from collections import defaultdict
from typing import Dict

import numpy as np
from ray.rllib.utils.annotations import override
from ray.rllib.utils.debug import update_global_seed_if_necessary
from ray.util.timer import _Timer

from fedlib.trainers import Trainer, TrainerConfig, TrainerCallback, TrainerCallbackList
from fedlib.clients import ClientConfig
from fedlib.constants import CLIENT_UPDATE, NUM_GLOBAL_STEPS, TRAIN_LOSS
from fedlib.core import WorkerGroupConfig, WorkerGroup, ClientWorkerGroup
from fedlib.datasets import DatasetCatalog
from fedlib.utils.types import PartialAlgorithmConfigDict


class FedavgCallback(TrainerCallback):
    def on_local_round_end(
        self,
        trainer: "FedavgTrainer",
    ):
        """Called when the local round ends."""


class FedavgTrainerCallbackList(TrainerCallbackList):
    def on_local_round_end(self, trainer: "FedavgTrainer"):
        """Called when the local round ends."""
        for callback in self._callbacks:
            callback.on_local_round_end(trainer)


class FedavgTrainerConfig(TrainerConfig):
    def __init__(self, algo_class=None):
        """Initializes a FedavgTrainerConfig instance."""
        super().__init__(algo_class=algo_class or FedavgTrainer)

        self.num_malicious_clients = 0

        self.callbacks_config = FedavgCallback

    def callbacks(self, callbacks_class) -> TrainerConfig:
        return super().callbacks(callbacks_class)

    def get_client_config(self) -> ClientConfig:
        config = ClientConfig(class_specifier="fedlib.clients.Client").update_from_dict(
            self.client_config
        )
        return config

    def get_worker_group_config(self) -> WorkerGroupConfig:
        if not self._is_frozen:
            raise ValueError(
                "Cannot call `get_learner_group_config()` on an unfrozen "
                "AlgorithmConfig! Please call `freeze()` first."
            )

        config = (
            WorkerGroupConfig(cls=ClientWorkerGroup)
            .resources(
                num_remote_workers=self.num_remote_workers,
                num_cpus_per_worker=self.num_cpus_per_worker,
                num_gpus_per_worker=self.num_gpus_per_worker,
            )
            .task(task_spec=self.get_task_spec())
        )
        return config

    @override(TrainerConfig)
    def build_callbacks(self, callbacllist_cls=None) -> TrainerCallbackList:
        return super().build_callbacks(callbacllist_cls or FedavgTrainerCallbackList)

    @override(TrainerConfig)
    def validate(self) -> None:
        super().validate()


class FedavgTrainer(Trainer):
    """Federated Averaging Algorithm."""

    def __init__(self, config=None, logger_creator=None, **kwargs):
        self.local_results = []
        super().__init__(config, logger_creator, **kwargs)

    @classmethod
    def get_default_config(cls) -> TrainerConfig:
        return FedavgTrainerConfig()

    def setup(self, config: TrainerConfig):
        # Set up our config: Merge the user-supplied config dict (which could
        # be a partial config dict) with the class' default.
        if not isinstance(config, TrainerConfig):
            assert isinstance(config, PartialAlgorithmConfigDict)
            config_obj = self.get_default_config()
            if not isinstance(config_obj, TrainerConfig):
                assert isinstance(config, PartialAlgorithmConfigDict)
                config_obj = TrainerConfig().from_dict(config_obj)
            config_obj.update_from_dict(config)
            self.config = config_obj

        # Set Algorithm's seed.
        update_global_seed_if_necessary("torch", self.config.random_seed)

        server_device = "cuda" if self.config.num_gpus_for_driver > 0 else "cpu"
        self.server = self.config.get_server_config().build(server_device)
        self.worker_group = self._setup_worker_group()

        self._setup_dataset()
        self.client_manager = self.client_manager_cls(
            self._dataset.client_ids,
            self._dataset.train_client_ids,
            self._dataset.test_client_ids,
            client_config=self.config.get_client_config(),
        )

        # Metrics-related properties.
        self._timers = defaultdict(_Timer)
        self._counters = defaultdict(int)
        self.global_vars = defaultdict(lambda: defaultdict(list))

        clients = self.client_manager.clients

        self.local_results = self.worker_group.foreach_execution(
            lambda _, client: client.setup(), clients
        )

        super().setup(config)

    def _setup_worker_group(self) -> WorkerGroup:
        worker_group_config = self.config.get_worker_group_config()
        worker_group = worker_group_config.build()
        return worker_group

    def _setup_dataset(self):
        self._dataset = DatasetCatalog.get_dataset(self.config.dataset_config)
        self.worker_group.setup_datasets(self._dataset)

    def training_step(self):
        self.worker_group.sync_weights(self.server.get_global_model().state_dict())

        def local_training(worker, client):
            dataset = worker.dataset.get_client_dataset(client.client_id)
            result = client.train_one_round(dataset)
            return result

        clients = self.client_manager.trainable_clients
        self.local_results = self.worker_group.foreach_execution(
            local_training, clients
        )

        self.callbacks.on_local_round_end(self)
        updates = [result.pop(CLIENT_UPDATE, None) for result in self.local_results]

        results = {}
        results.update(self.compile_train_results(self.local_results))

        self._counters[NUM_GLOBAL_STEPS] += 1
        global_vars = {
            "timestep": self._counters[NUM_GLOBAL_STEPS],
        }

        server_return = self.server.step(updates, global_vars)

        results.update(server_return)
        return results

    def compile_train_results(self, results):
        losses = []
        for result in results:
            loss = result.pop(TRAIN_LOSS, None)
            if loss is None:
                return {}
            losses.append(loss)
        results = {TRAIN_LOSS: np.mean(losses)}
        return results

    def evaluate(self):
        self.worker_group.sync_weights(self.server.get_global_model().state_dict())
        clients = self.client_manager.testable_clients

        def validate_func(worker, client):
            client_dataset = worker.dataset.get_client_dataset(client.client_id)
            test_loader = client_dataset.get_test_loader()
            return client.evaluate(test_loader)

        evaluate_results = self.worker_group.foreach_execution(validate_func, clients)
        results = self.server.task.compile_eval_results(evaluate_results)

        return results

    def save_checkpoint(self, checkpoint_dir: str) -> Dict | None:
        pass
