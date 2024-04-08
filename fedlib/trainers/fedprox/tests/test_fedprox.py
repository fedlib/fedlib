import copy
import unittest

import ray
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fedlib.trainers.fedprox import FedProxTrainerConfig
from fedlib.datasets import DatasetCatalog
from fedlib.datasets import ToyFLDataset


class TestFedprox(unittest.TestCase):
    def setUp(self):
        DatasetCatalog.register_custom_dataset("simple", ToyFLDataset)
        model = torch.nn.Linear(2, 2)

        self.global_lr = 0.1
        self.alg = (
            FedProxTrainerConfig()
            .resources(num_remote_workers=2, num_gpus_per_worker=0)
            .data(
                num_clients=1,
                dataset_config={
                    "custom_dataset": "simple",
                },
            )
            .training(
                global_model=model,
                server_config={"lr": self.global_lr, "aggregator": {"type": "Mean"}},
            )
            .client(client_config={"lr": 1.0})
            .build()
        )
        self.global_dataset = DatasetCatalog.get_dataset(
            {
                "custom_dataset": "simple",
            },
        )

    @classmethod
    def tearDownClass(cls):
        ray.shutdown()

    def test_on_local_round_end(self):
        train_set, _ = self.global_dataset.to_torch_datasets()

        for _ in range(5):
            for data, target in DataLoader(dataset=train_set, batch_size=3):
                model = copy.deepcopy(self.alg.server.get_global_model())
                opt = torch.optim.SGD(model.parameters(), lr=0.1)
                model.train()
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                opt.step()

            self.alg.training_step()
            updated_model = copy.deepcopy(self.alg.server.get_global_model())
            self.assertTrue(torch.allclose(model.weight, updated_model.weight))

    def test_evaluate(self):
        train_loader = self.alg._dataset.server_dataset
        data, target = train_loader.get_next_train_batch()

        self.alg.training_step()
        task = self.alg.server.task
        task.init_optimizer()
        task.train_one_batch(data, target)
        result = self.alg.evaluate()
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
