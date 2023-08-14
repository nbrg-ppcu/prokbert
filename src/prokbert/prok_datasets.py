# module for datasets
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import torch
import numpy as np
import h5py
from typing import Dict, List, Type, Tuple, Iterator
from torch.utils.data import Dataset, IterableDataset

class IterableProkBERTPretrainingDataset(IterableDataset):
    def __init__(self, file_path: str, 
                 input_batch_size: int = 10000,
                 ds_offset: int = 0,
                 max_iteration_over_ds: int = 10) -> None:
        """
        Initialize the IterableProkBERTPretrainingDataset.

        :param file_path: Path to the HDF5 file.
        :param input_batch_size: Batch size for data fetching.
        :param ds_offset: Offset for data fetching.
        :param max_iteration_over_ds: Maximum number of iterations over the dataset.

        Example:
            >>> dataset = RefactoredIterableProkBERTPretrainingDataset(file_path="path/to/file.hdf5")
            >>> for data in dataset:
            >>>     # process data
        """
        self.file_path = file_path
        self.input_batch_size = input_batch_size
        self.ds_offset = ds_offset

        with h5py.File(self.file_path, 'r') as dataset_file:
            self.ds_size = dataset_file['training_data']['X'].shape[0]

        self.max_iteration_over_ds_steps = int(self.ds_size * max_iteration_over_ds)
        self._global_iter_steps = 0
        logging.info(f'Dataset size: {self.ds_size}')

    def __len__(self) -> int:
        return self.ds_size

    def _get_fetch_interval(self) -> Tuple[int, int]:
        max_ds_items = self.ds_size
        new_fetch_start = self._current_ds_pointer * self.input_batch_size
        new_fetch_end = (self._current_ds_pointer + 1) * self.input_batch_size

        if new_fetch_end > max_ds_items:
            new_fetch_end = max_ds_items

        if new_fetch_start >= max_ds_items - 1:
            raise StopIteration

        return new_fetch_start, new_fetch_end

    def _fetch_new_data(self) -> None:
        new_fetch_start, new_fetch_end = self._get_fetch_interval()
        
        with h5py.File(self.file_path, 'r') as dataset_file:
            self._current_data_batch = torch.tensor(dataset_file['training_data']['X'][new_fetch_start:new_fetch_end],
                                                    dtype=torch.int)

    def __iter__(self) -> Iterator[torch.Tensor]:
        self._current_ds_pointer = int(np.floor(self.ds_offset / self.input_batch_size))
        self._current_data_pointer = 0
        self._fetch_new_data()
        self._global_iter_steps = 0
        return self

    def __next__(self) -> torch.Tensor:
        try:
            ds_item = self._current_data_batch[self._current_data_pointer]
            self._current_data_pointer += 1
        except IndexError:
            self._fetch_new_data()
            self._current_data_pointer = 0
            ds_item = self._current_data_batch[self._current_data_pointer]
            self._current_data_pointer += 1

        self._global_iter_steps += 1
        if self._global_iter_steps > self.max_iteration_over_ds_steps:
            logging.info('Stopping the iteration.')
            raise StopIteration

        return ds_item
    

class IterableProkBERTPretrainingDatasetOld(IterableDataset):

    def __init__(self, file_path: str, 
                input_batch_size: int = 10000,
                ds_offset: int = 0,
                max_iteration_over_ds = 10):
        """
        Initializer for IterableProkBERTPretrainingDataset.
        """
        self.file_path = file_path
        self.input_batch_size = input_batch_size
        self.ds_offset = ds_offset

        with h5py.File(self.file_path, 'r') as dataset_file:
            self.ds_size = dataset_file['training_data']['X'].shape[0]

        self.max_iteration_over_ds_steps = int(self.ds_size*max_iteration_over_ds)
        self._global_iter_steps = 0

        logging.info(f'Dataset size: {self.ds_size}')

    def __len__(self) -> int:
        return self.ds_size

    def _get_fetch_interval(self) -> Tuple[int, int]:
        """
        Get the interval for fetching the next batch from the HDF5 file.
        """
        print('Get next interval info')
        max_ds_items = self.ds_size
        new_fetch_start = self._act_ds_pointer * self.input_batch_size
        new_fetch_end = (self._act_ds_pointer + 1) * self.input_batch_size
        print(f'Fetched intervals: new_fetch_start: {new_fetch_start} new_fetch_end: {new_fetch_end}')
        if new_fetch_end > max_ds_items:
            new_fetch_end = max_ds_items
        if new_fetch_start >= max_ds_items - 1:
            raise StopIteration
        return new_fetch_start, new_fetch_end

    def _fetch_new_data(self):
        """
        Fetch new data from the HDF5 file.
        """
        try:
            new_fetch_start, new_fetch_end = self._get_fetch_interval()
        except StopIteration:
            logging.info('Fetching new data')
            self._act_ds_pointer = 0
            self._act_ds_pointer_it = 0
            new_fetch_start, new_fetch_end = self._get_fetch_interval()
            
        with h5py.File(self.file_path, 'r') as dataset_file:
            self._act_torchds = torch.tensor(dataset_file['training_data']['X'][new_fetch_start:new_fetch_end],
                                         dtype=torch.int)

    def __iter__(self) -> Iterator[torch.Tensor]:
        """
        Create an iterator for the dataset.
        """
        self._act_ds_pointer = int(np.floor(self.ds_offset / self.input_batch_size))
        print(f'Pointer: {self._act_ds_pointer}')
        self._act_ds_pointer_it = 0
        self._fetch_new_data()
        self._global_iter_steps = 0

        return self

    def __next__(self) -> torch.Tensor:
        """
        Fetch the next item from the dataset.
        """
        try:
            ds_item = self._act_torchds[self._act_ds_pointer_it]
            self._act_ds_pointer_it += 1
        except IndexError:
            self._fetch_new_data()
            self._act_ds_pointer_it = 0
            ds_item = self._act_torchds[self._act_ds_pointer_it]
            self._act_ds_pointer_it += 1
        self._global_iter_steps+=1
        if self._global_iter_steps > self.max_iteration_over_ds_steps:
            print('Stoppfing the iteration!')
            raise(StopIteration)

        return ds_item