# module for datasets
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import torch
import numpy as np
import h5py
from typing import Dict, List, Type, Tuple, Iterator, Union
from torch.utils.data import Dataset, IterableDataset
import torch.distributed as dist

#class IterableProkBERTPretrainingDataset(IterableDataset):

class IterableProkBERTPretrainingDataset(IterableDataset):
    def __init__(self, file_path: str, 
                 input_batch_size: int = 10000,
                 ds_offset: int = 0,
                 max_iteration_over_ds: int = 10,
                 default_dtype = torch.long,
                 add_end_token=False):
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
        self._global_iter_steps = 0,
        self.default_dtype = default_dtype
        logging.info(f'Dataset size: {self.ds_size}')
        self.add_end_token = add_end_token
                # Assuming you're using distributed training
        self.rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


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
                                         dtype=self.default_dtype)
        if self.add_end_token:
            threes_column = torch.full((self._current_data_batch.shape[0], 1), 3, dtype=self.default_dtype)
            self._current_data_batch = torch.cat((self._current_data_batch, threes_column), dim=1)

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
            data = torch.tensor(self.dataset_file['training_data']['X'][index], dtype=self.default_dtype)
            if self.add_end_token:
                threes_column = torch.tensor([3], dtype=self.default_dtype)
                data =  torch.cat((data, threes_column), dim=0)                
            return data
        elif isinstance(index, slice):
            # Return slice
            data = torch.tensor(self.dataset_file['training_data']['X'][index], dtype=self.default_dtype)
            if self.add_end_token:
                threes_column = torch.full((data.shape[0], 1), 3, dtype=self.default_dtype)
                data = torch.cat((data, threes_column), dim=1)
            return data
            #return [torch.tensor(item, dtype=torch.int) for item in self.dataset_file['training_data']['X'][index]]

    
class ProkBERTPretrainingDatasetPT(Dataset):
    def __init__(self, filepath):
        print(f'Loading: {filepath}')
        self.X = torch.load(filepath)
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # 
        return self.X[index]    

class ProkBERTPretrainingDataset(Dataset):
    def __init__(self, X):
        self.X = X
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # 
        return torch.tensor(self.X[index], dtype=torch.long)
    

class ProkBERTPretrainingHDFDataset(Dataset):
    def __init__(self, hdf_file_path: str, default_dtype = torch.long):
        """
        Initialize the HDFDataset.

        :param hdf_file_path: Path to the HDF5 file.
        """
        self.default_dtype = default_dtype
        self.file_path = hdf_file_path
        self.dataset_file = h5py.File(self.file_path, 'r')
        logging.info(f'Loading and converting file {hdf_file_path}')
        if self.dataset_file['training_data']['X'].dtype == 'uint16':
            #print('Its an uint dataset converting first')
            self.dataset = np.array(self.dataset_file['training_data']['X']).astype(np.int32)
            self.dataset = torch.tensor(self.dataset, dtype = self.default_dtype)

        else:
            self.dataset = torch.tensor(np.array(self.dataset_file['training_data']['X']), dtype = self.default_dtype)
        logging.info(f'Loading finished!')

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
        #self._ensure_file_open()
        if isinstance(index, int):
            # Return single item
            return self.dataset[index]
        elif isinstance(index, slice):
            # Return slice
            return self.dataset[index]

    def __len__(self):
        #self._ensure_file_open()
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
        pass
        #self.close()

class TestDS(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data  # This should be a list of dictionaries

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    

class ProkBERTTrainingDatasetPT(Dataset):
    def __init__(self, X, y, attention_masks=None, AddAttentionMask=False):
        self.input_ids = X  # Assuming X is a tensor containing input_ids
        self.labels = y  # Assuming y is a tensor containing labels
        self.attention_masks = attention_masks  # Optional attention masks
        self.AddAttentionMask = AddAttentionMask

    def __len__(self):
        return len(self.labels)  # Number of samples

    def __getitem__(self, idx):
        sample = {
            'input_ids': self.input_ids[idx],  # input_ids for this sample
            'labels': self.labels[idx],  # label for this sample
            
        }
        if self.AddAttentionMask:
            attention_mask = (self.input_ids[idx] > 3) |  (self.input_ids[idx] == 2) | (self.input_ids[idx] == 1)
            attention_mask = attention_mask.float()
            #sample['attention_mask'] = (self.input_ids[idx] != 0).float()
            sample['attention_mask'] = attention_mask

        # Include attention_mask in the sample if it is provided
        if self.attention_masks is not None:
            sample['attention_mask'] = self.attention_masks[idx]
        
        return sample


class ProkBERTTrainingDatasetPTa(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X  # Assuming X is either a list of dictionaries or a tensor
        self.y = y  # Assuming y is a tensor of labels

    def __len__(self):
        return len(self.y)  # Assuming y is a tensor, so this gives us the number of samples

    def __getitem__(self, idx):
        x = self.X[idx]  # Get the data sample at the given index
        y = self.y[idx]  # Get the corresponding label

        return x, y  # Return the sample and its label

    

      