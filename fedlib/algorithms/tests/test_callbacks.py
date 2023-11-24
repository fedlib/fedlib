import unittest

import ray
import torch

from fedlib.algorithms import AlgorithmCallback
from fedlib.algorithms.fedavg import FedavgConfig
from fedlib.datasets import DatasetCatalog, ToyFLDataset


class InitCallbacks(AlgorithmCallback):
    def on_algorithm_init(self, *, algorithm, **kwargs):
        self._on_init_was_called = True


class TestCallbacks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ray.init()
        DatasetCatalog.register_custom_dataset("simple", ToyFLDataset)

    @classmethod
    def tearDownClass(cls):
        ray.shutdown()

    def test_on_algorithm_init(self):
        model = torch.nn.Linear(2, 2)
        algo = (
            FedavgConfig()
            .resources(num_remote_workers=2, num_gpus_per_worker=0)
            .data(
                num_clients=1,
                dataset_config={"custom_dataset": "simple"},
            )
            .training(global_model=model)
            .callbacks(InitCallbacks)
            .build()
        )

        self.assertTrue(algo.callbacks[0]._on_init_was_called)
