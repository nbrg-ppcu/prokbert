import unittest
import torch
from transformers import BertTokenizer
import numpy as np

import unittest
import os
import numpy as np
import h5py
import torch

from prokbert.prok_datasets import *


class TestProkBERTDatasets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_file = "data/test_dataset.hdf5"
        with h5py.File(cls.test_file, 'w') as f:
            data = np.random.randint(0, 256, size=(100, 128), dtype=np.int16)
            f.create_dataset("training_data/X", data=data)

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.test_file)

    def test_iterable_dataset_initialization(self):
        dataset = IterableProkBERTPretrainingDataset(self.test_file, input_batch_size=10)
        self.assertIsNotNone(dataset)
        dataset.__del__()

    def test_iterable_dataset_iteration(self):
        dataset = IterableProkBERTPretrainingDataset(self.test_file, input_batch_size=10)
        for _ in dataset:
            pass
        dataset.__del__()

    def test_iterable_dataset_item_retrieval(self):
        dataset = IterableProkBERTPretrainingDataset(self.test_file, input_batch_size=10)
        item = dataset[10]
        self.assertTrue(isinstance(item, torch.Tensor))
        items = dataset[10:20]
        self.assertTrue(isinstance(items, list))
        dataset.__del__()

    def test_simple_dataset(self):
        X = np.random.randint(0, 256, size=(100, 128), dtype=np.int16)
        dataset = ProkBERTPretrainingDataset(X)
        self.assertEqual(len(dataset), 100)
        item = dataset[10]
        self.assertTrue(isinstance(item, torch.Tensor))

    def test_hdf_dataset_initialization(self):
        dataset = ProkBERTPretrainingHDFDataset(self.test_file)
        self.assertIsNotNone(dataset)

    def test_hdf_dataset_item_retrieval(self):
        dataset = ProkBERTPretrainingHDFDataset(self.test_file)
        item = dataset[10]
        self.assertTrue(isinstance(item, torch.Tensor))

    def test_hdf_dataset_slice_retrieval(self):
        dataset = ProkBERTPretrainingHDFDataset(self.test_file)
        items = dataset[10:20]
        self.assertTrue(isinstance(items, torch.Tensor))
        self.assertEqual(items.shape[0], 10)

    def test_hdf_dataset_length(self):
        dataset = ProkBERTPretrainingHDFDataset(self.test_file)
        self.assertEqual(len(dataset), 100)

    def test_hdf_dataset_reopen(self):
        dataset = ProkBERTPretrainingHDFDataset(self.test_file)
        dataset.close()
        item = dataset[10]  # This should reopen the file automatically
        self.assertTrue(isinstance(item, torch.Tensor))
        dataset.close()


if __name__ == '__main__':
    unittest.main()