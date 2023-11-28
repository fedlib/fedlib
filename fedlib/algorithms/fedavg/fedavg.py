from collections import defaultdict
from typing import Dict

import numpy as np
from ray.rllib.utils.annotations import override
from ray.rllib.utils.debug import update_global_seed_if_necessary
from ray.util.timer import _Timer

from fedlib.algorithms import Algorithm, AlgorithmConfig
from fedlib.clients import ClientConfig
from fedlib.constants import CLIENT_UPDATE, NUM_GLOBAL_STEPS
from fedlib.core import WorkerGroupConfig, WorkerGroup, ClientWorkerGroup
from fedlib.datasets import FLDataset, DatasetCatalog
from fedlib.types import PartialAlgorithmConfigDict


class FedavgConfig(AlgorithmConfig):
    def __init__(self, algo_class=None):
        """Initializes a FedavgConfig instance."""
        super().__init__(algo_class=algo_class or Fedavg)

        self.num_malicious_clients = 0

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

    @override(AlgorithmConfig)
    def validate(self) -> None:
        super().validate()

        # This condition is only for DnC and Trimmedmean aggregators.
        if (
            self.server_config.get("aggregator", None) is not None
            and self.server_config.get("aggregator").get("type") is not None
            # and "num_byzantine" not in self.server_config["aggregator"]["type"]
            and (
                "DnC" == self.server_config["aggregator"]["type"]
                or "Trimmedmean" in self.server_config["aggregator"]["type"]
                or "Multikrum" in self.server_config["aggregator"]["type"]
            )
        ):
            self.server_config["aggregator"][
                "num_byzantine"
            ] = self.num_malicious_clients

        # Check whether the number of malicious clients makes sense.
        if self.num_malicious_clients > self.num_clients:
            raise ValueError(
                "`num_malicious_clients` must be smaller than or equal "
                "`num_clients`! Simulation makes no sense otherwise."
            )


class Fedavg(Algorithm):
    """Federated Averaging Algorithm."""

    def __init__(self, config=None, logger_creator=None, **kwargs):
        self.local_results = []
        super().__init__(config, logger_creator, **kwargs)

    @classmethod
    def get_default_config(cls) -> AlgorithmConfig:
        return FedavgConfig()

    def setup(self, config: AlgorithmConfig):
        super().setup(config)
        # Set up our config: Merge the user-supplied config dict (which could
        # be a partial config dict) with the class' default.
        if not isinstance(config, AlgorithmConfig):
            assert isinstance(config, PartialAlgorithmConfigDict)
            config_obj = self.get_default_config()
            if not isinstance(config_obj, AlgorithmConfig):
                assert isinstance(config, PartialAlgorithmConfigDict)
                config_obj = AlgorithmConfig().from_dict(config_obj)
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
            if isinstance(worker.dataset, FLDataset):
                dataset = worker.dataset.get_client_dataset(client.client_id)
            else:
                dataset = worker.dataset.get_train_loader(client.client_id)
            return client.train_one_round(dataset)

        clients = self.client_manager.trainable_clients
        self.local_results = self.worker_group.foreach_execution(
            local_training, clients
        )

        updates = [result.pop(CLIENT_UPDATE, None) for result in self.local_results]
        losses = []
        for result in self.local_results:
            loss = result.pop("avg_loss")
            losses.append(loss)

        self._counters[NUM_GLOBAL_STEPS] += 1
        # train_results
        global_vars = {
            "timestep": self._counters[NUM_GLOBAL_STEPS],
        }
        results = {"train_loss": np.mean(losses)}
        server_return = self.server.step(updates, global_vars)
        results.update(server_return)

        return results

    def evaluate(self):
        self.worker_group.sync_weights(self.server.get_global_model().state_dict())
        clients = self.client_manager.testable_clients

        def validate_func(worker, client):
            # test_loader = worker.dataset.get_test_loader(client.client_id)
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

    def save_checkpoint(self, checkpoint_dir: str) -> Dict | None:
        pass
