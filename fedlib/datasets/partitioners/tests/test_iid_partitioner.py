import unittest

import torch
from torch.utils.data import Dataset

# Assuming IIDPartitioner is defined in the fedlib.keyconcepts.partitioners module
from fedlib.datasets.partitioners import IIDPartitioner


# Mock Dataset
class MockDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = torch.randn(size, 10)  # Mock data

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


class TestIIDPartitioner(unittest.TestCase):
    def test_split_dataset(self):
        dataset_size = 100
        num_clients = 7

        # Create mock dataset
        dataset = MockDataset(dataset_size)

        # Initialize IID partitioner
        partitioner = IIDPartitioner(num_clients=num_clients)

        # Partition the dataset
        subsets = partitioner.split_dataset(dataset)

        # Check that we have the correct number of subsets
        self.assertEqual(len(subsets), num_clients)

        # Check that each client has at least one sample
        for subset in subsets:
            self.assertGreater(len(subset), 0)

        # Check that all samples are allocated
        subset_size = sum([len(subset) for subset in subsets])
        self.assertEqual(subset_size, dataset_size)

    def test_split_datasets(self):
        dataset_size = 100
        num_clients = 5

        # Create mock keyconcepts for training and testing
        train_dataset = MockDataset(dataset_size)
        test_dataset = MockDataset(dataset_size)

        # Initialize IID partitioner
        partitioner = IIDPartitioner(num_clients=num_clients)

        # Partition the keyconcepts
        paired_subsets = partitioner.generate_paired_subsets(
            train_dataset, test_dataset
        )
        # Check that we have the correct number of paired subsets
        self.assertEqual(len(paired_subsets), num_clients)

        # Check that each client has at least one sample for both training and testing
        for train_subset, test_subset in paired_subsets.values():
            self.assertGreater(len(train_subset), 0)
            self.assertGreater(len(test_subset), 0)


if __name__ == "__main__":
    unittest.main()
