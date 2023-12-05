import copy
from statistics import mean
from typing import Iterable, Dict, Optional, List, Callable

import numpy as np
import ray
import torch
import torch.nn.functional as F
from ray import tune
from ray.rllib.utils.typing import (
    ResultDict,
)
from ray.tune.stopper import MaximumIterationStopper
from torch import Tensor
from torch import nn
from torch.autograd import Variable

from fedlib.algorithms import AlgorithmConfig, Server
from fedlib.algorithms.fedavg import FedavgConfig, Fedavg
from fedlib.algorithms.server_config import ServerConfig
from fedlib.clients import Client, ClientConfig
from fedlib.constants import NUM_GLOBAL_STEPS
from fedlib.core import WorkerGroupConfig
from fedlib.core.execution.session import get_session
from fedlib.core.execution.worker import Worker
from fedlib.datasets import FLDataset
from fedlib.tasks import Classifier, TaskSpec


class SplitClient(Client):
    def train_one_round(
        self,
        data_reader: Iterable,
    ):
        self._train_round += 1
        sess = get_session(self.client_id)
        sess.task.zero_psudo_grad()
        data, target = data_reader.get_next_train_batch()
        data, target = self.callbacks.on_train_batch_begin(data, target)
        smashed_data = sess.task.train_one_batch(
            data, target, self.callbacks.on_backward_end
        )
        return {"smashed_data": smashed_data, "labels": target}


class MLPHeaD(nn.Module):
    def __init__(self):
        super(MLPHeaD, self).__init__()
        hidden_1 = 128
        hidden_2 = 256
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class SplitClassificationServer(Classifier):
    def make_model(self) -> torch.nn.Module:
        return MLPHeaD()


class SplitServer(Server):
    def step(
        self,
        local_updates: List[Tensor],
        global_vars: Optional[Dict[str, torch.Tensor]] = None,
    ) -> ResultDict:
        updates = local_updates
        model = self.get_global_model()
        model.train()
        smashed_grads = []
        losses = []
        for update in updates:
            model.zero_grad()
            smashed_data: torch.Tensor = update["smashed_data"]
            target = update["labels"]
            head_input = Variable(smashed_data, requires_grad=True)
            loss = self.task.loss(model, head_input, target)
            loss.backward()
            smashed_grads.append(head_input.grad.data)
            losses.append(loss.item())

            for opt in self._optimizers:
                opt.step()

        return {"smached_grads": smashed_grads, "loss": mean(losses)}


class SplitClassificationClient(Classifier):
    def make_model(self) -> torch.nn.Module:
        class MLPClient(nn.Module):
            def __init__(self):
                super(MLPClient, self).__init__()
                hidden_1 = 128
                self.fc1 = nn.Linear(28 * 28, hidden_1)
                self.dropout = nn.Dropout(0.2)

            def forward(self, x):
                x = x.view(-1, 28 * 28)
                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                return x

        client_model = MLPClient()

        class FullModel(nn.Module):
            def __init__(self, modelA, modelB):
                super(FullModel, self).__init__()
                self.modelA = modelA
                self.modelB = modelB

            def forward(self, x):
                x = self.modelA(x)
                x = self.modelB(x)
                return x

        self.head_model = MLPHeaD()
        self.full_model = FullModel(client_model, self.head_model)
        return self.full_model
        # return MLPClient()

    def evaluate(self, test_loader):
        if self._metrics is None:
            self._metrics = self.init_metrics()
        device = next(self.full_model.parameters()).device
        self._metrics.to(device)
        self._metrics.reset()
        self._model.eval()

        result = {"length": 0}
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = self.full_model(data)
                result["length"] += len(target)
                self._metrics(output, target)
        result.update(self._metrics.compute())
        return result

    def train_one_batch(
        self,
        data: torch.Tensor,
        target: torch.Tensor,
        on_backward_end: Callable = None,
    ):
        model = self.full_model.modelA
        model.train()
        # Get the device on which the model is located
        device = next(model.parameters()).device
        self.data, self.target = data.to(device), target.to(device)
        self.output: Tensor = model(data)

        return self.output

    def client_backpropagation(self, output_grad):
        model = self.full_model.modelA
        model.train()
        for _, opt in enumerate(self._optimizers):
            opt.zero_grad()
        self.output: Tensor = model(self.data)
        self.output.backward(output_grad.clone().detach().to(self.output.device))
        for _, opt in enumerate(self._optimizers):
            opt.step()


class ExampleSplitnnConfig(FedavgConfig):
    def __init__(self, algo_class=None):
        """Initialize a FedavgConfig instance."""
        super().__init__(algo_class=algo_class or ExampleSplit)

        self.dataset_config = {
            "type": "MNIST",
            "num_clients": 1,
            "train_batch_size": 32,
        }
        self.global_model = "cnn"

    def get_server_config(self) -> ServerConfig:
        if not self._is_frozen:
            raise ValueError(
                "Cannot call `get_server_config()` on an unfrozen "
                "AlgorithmConfig! Please call `freeze()` first."
            )

        config = ServerConfig(
            class_specifier=SplitServer,
            task_spec=TaskSpec(task_class=SplitClassificationServer, alg_config=self),
        ).update_from_dict(self.server_config)
        return config

    def get_client_config(self) -> ClientConfig:
        config = ClientConfig(class_specifier=SplitClient).update_from_dict(
            self.client_config
        )
        return config

    def get_worker_group_config(self) -> WorkerGroupConfig:
        class SplitNNWorker(Worker):
            def switch_client(self, client_id):
                if self._current_client_id is not None:
                    self._client_states[self._current_client_id] = [
                        copy.deepcopy(self.task.get_optimizer_states()),
                        copy.deepcopy(self.task.get_model_states()),
                    ]
                self._current_client_id = client_id
                opt_states, model_states = self._client_states.get(
                    client_id, [None, None]
                )
                if opt_states is not None:
                    self.task.set_optimizer_states(opt_states)
                    self.task.set_model_states(model_states)

        config = super().get_worker_group_config()
        return config.task(
            TaskSpec(task_class=SplitClassificationClient, alg_config=self)
        ).worker(worker_class=SplitNNWorker)


class ExampleSplit(Fedavg):
    def __init__(self, config=None, logger_creator=None, **kwargs):
        super().__init__(config, logger_creator, **kwargs)

    @classmethod
    def get_default_config(cls) -> AlgorithmConfig:
        return ExampleSplitnnConfig()

    def training_step(self):
        def local_training(worker, client):
            if isinstance(worker.dataset, FLDataset):
                dataset = worker.dataset.get_client_dataset(client.client_id)
            else:
                dataset = worker.dataset.get_train_loader(client.client_id)
            return client.train_one_round(dataset)

        clients = self.client_manager.trainable_clients
        results = self.worker_group.foreach_execution(local_training, clients)
        local_results = [result for result in results]

        self._counters[NUM_GLOBAL_STEPS] += 1
        # train_results
        global_vars = {
            "timestep": self._counters[NUM_GLOBAL_STEPS],
        }
        results = {}
        server_return = self.server.step(local_results, global_vars)
        grads = server_return.pop("smached_grads")

        for client, grad in zip(clients, grads):
            client.output_grad = grad.clone().detach()

        def client_backpropagation(worker, client):
            sess = get_session(client.client_id)
            sess.task.client_backpropagation(client.output_grad)

        _ = self.worker_group.foreach_execution(client_backpropagation, clients)
        results.update(server_return)
        print(results)
        return results

    def evaluate(self):
        self.worker_group.sync_state(
            "head_model", self.server.get_global_model().state_dict()
        )
        clients = self.client_manager.testable_clients

        def validate_func(worker, client):
            head_model_state = worker.get_state("head_model")

            worker.task._model.modelB.load_state_dict(head_model_state)
            if isinstance(worker.dataset, FLDataset):
                test_loader = worker.dataset.get_client_dataset(
                    client.client_id
                ).get_test_loader()
            else:
                test_loader = worker.dataset.get_test_loader(client.client_id)
            return client.evaluate(test_loader)

        evaluate_results = self.worker_group.foreach_execution(validate_func, clients)

        results = {
            "ce_loss": np.average(
                [metric["ce_loss"] for metric in evaluate_results],
                weights=[metric["length"] for metric in evaluate_results],
            ),
            "acc_top_1": np.average(
                [metric["acc_top_1"].cpu() for metric in evaluate_results],
                weights=[metric["length"] for metric in evaluate_results],
            ),
        }
        return results


if __name__ == "__main__":
    ray.init()

    config_dict = (
        ExampleSplitnnConfig()
        .resources(
            num_gpus_for_driver=0.0,
            num_cpus_for_driver=1,
            num_remote_workers=0,
            num_gpus_per_worker=0.0,
        )
        .evaluation(evaluation_interval=200)
        .to_dict()
    )
    print(config_dict)
    tune.run(
        ExampleSplit,
        config=config_dict,
        stop=MaximumIterationStopper(200),
    )
