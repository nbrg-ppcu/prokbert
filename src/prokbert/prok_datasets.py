# module for datasets
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import torch
import numpy as np
import h5py
from typing import Dict, List, Type, Tuple, Iterator, Union
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
        self.dataset_file = h5py.File(self.file_path, 'r')
        self.ds_size = self.dataset_file['training_data']['X'].shape[0]
        self.max_iteration_over_ds_steps = int(self.ds_size * max_iteration_over_ds)
        self._global_iter_steps = 0
        logging.info(f'Dataset size: {self.ds_size}')

    def __len__(self) -> int:
        return self.ds_size

    def _ensure_file_open(self):
        """
        Ensure the HDF5 file is open. If not, reopen it.
        """
        try:
            # Try accessing the 'mode' attribute to check if file is open
            _ = self.dataset_file.mode
        except ValueError as e:
            # If we catch a ValueError, it means the file might be closed. We confirm by checking the error message.
            self.dataset_file = h5py.File(self.file_path, 'r')

    def _get_fetch_interval(self) -> Tuple[int, int]:
        max_ds_items = self.ds_size
        new_fetch_start = self._current_ds_pointer * self.input_batch_size
        new_fetch_end = (self._current_ds_pointer + 1) * self.input_batch_size

        if new_fetch_end > max_ds_items:
            new_fetch_end = max_ds_items

        if new_fetch_start >= max_ds_items - 1:
            raise StopIteration

        return new_fetch_start, new_fetch_end

    def _fetch_new_data(self):
        """
        Fetch new data from the HDF5 file.
        """
        # Check if the file is still open, if not, reopen it
        self._ensure_file_open()

        new_fetch_start, new_fetch_end = self._get_fetch_interval()
        self._current_data_batch = torch.tensor(self.dataset_file['training_data']['X'][new_fetch_start:new_fetch_end],
                                         dtype=torch.int32)
        

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
    
    def __getitem__(self, index) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Get item or slice from the dataset.

        :param index: Index or slice object
        :return: Dataset item or slice
        """
        # Ensure the file is open
        self._ensure_file_open()

        if isinstance(index, int):
            # Return single item
            return torch.tensor(self.dataset_file['training_data']['X'][index], dtype=torch.int32)
        elif isinstance(index, slice):
            # Return slice
            return torch.tensor(self.dataset_file['training_data']['X'][index], dtype=torch.int32)
        
            #return [torch.tensor(item, dtype=torch.int) for item in self.dataset_file['training_data']['X'][index]]


    def __del__(self):
        """
        Destructor to close the HDF5 file when the dataset object is destroyed.
        """
        self.dataset_file.close()
    
    

class ProkBERTPretrainingDataset(Dataset):
    def __init__(self, X):
        self.X = X
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # 
        return torch.tensor(self.X[index], dtype=torch.int16)

class ProkBERTPretrainingHDFDataset(Dataset):
    def __init__(self, hdf_file_path: str):
        """
        Initialize the HDFDataset.

        :param hdf_file_path: Path to the HDF5 file.
        """
        self.file_path = hdf_file_path
        self.dataset_file = h5py.File(self.file_path, 'r')
        self.dataset = self.dataset_file['training_data']['X']

    def _ensure_file_open(self):
        """
        Ensure the HDF5 file is open. If not, reopen it.
        """
        try:
            # Try accessing the 'mode' attribute to check if file is open
            _ = self.dataset_file.mode
        except ValueError as e:
            # If we catch a ValueError, it means the file might be closed. Reopen it.
            self.dataset_file = h5py.File(self.file_path, 'r')
            self.dataset = self.dataset_file['training_data']['X']

    def __getitem__(self, index):
        """
        Fetch data using an index or a slice.

        :param index: Index or slice object
        :return: Dataset item or slice
        """
        self._ensure_file_open()
        if isinstance(index, int):
            # Return single item
            return torch.tensor(self.dataset[index], dtype=torch.int16)
        elif isinstance(index, slice):
            # Return slice
            return torch.tensor(self.dataset[index], dtype=torch.int16)

    def __len__(self):
        self._ensure_file_open()
        return len(self.dataset)

    def close(self):
        """
        Close the HDF5 file.
        """
        self.dataset_file.close()

    def __del__(self):
        """
        Destructor to close the HDF5 file when the dataset object is destroyed.
        """
        self.close()
