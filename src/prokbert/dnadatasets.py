# coding=utf-8
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, IterableDataset
import torch
import h5py
import logging
import os
import pickle
import random
import re
import shutil
import numpy as np
import math

from transformers import (
    PreTrainedTokenizer
)

class ProkDataset(torch.utils.data.Dataset):
    """
    A custom PyTorch Dataset for processing tokenized sequences and labels.
    
    """    
    def __init__(self, tokenized_seqs, labels, token_type_ids=None, attention_mask=None, device=None):
        """
        Constructor

        :param tokenized_seqs: Tokenized sequences represented as a tensor.
        :type tokenized_seqs: torch.Tensor

        :param labels: Labels corresponding to each tokenized sequence.
        :type labels: torch.Tensor

        :param token_type_ids: Token type IDs corresponding to each tokenized sequence. Defaults to None.
        :type token_type_ids: torch.Tensor, optional

        :param attention_mask: Attention mask corresponding to each tokenized sequence. Defaults to None.
        :type attention_mask: torch.Tensor, optional

        :param device: Device to move the tensors onto. If provided, the tensors are moved to the specified device.
                    Defaults to None.
        :type device: str, optional
        """
        self.tokenized_seqs = tokenized_seqs
        self.labels = labels
        if attention_mask is not None:
            self.attention_mask = attention_mask
        if token_type_ids is not None:
            self.token_type_ids = token_type_ids
        if device is not None:
            self.tokenized_seqs.to(device)
            self.labels.to(device)
            self.attention_mask.to(device)
            self.token_type_ids.to(device)

    def __getitem__(self, idx):
        """
        Get the data item at the specified index.

        :param idx: Index of the item to retrieve.
        :type idx: int

        :return: A dictionary containing 'input_ids', 'labels', 'token_type_ids', and 'attention_mask' tensors.
        :rtype: dict
        """
        #item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item={'input_ids' : self.tokenized_seqs[idx,:]}
        item['labels'] = self.labels[idx]
        item['token_type_ids'] = self.token_type_ids[idx,:]
        item['attention_mask'] = self.attention_mask[idx,:]
        
        return item
    
    def __len__(self):
        """
        Get the total number of samples in the dataset.

        :return: The total number of samples in the dataset.
        :rtype: int
        """
        return len(self.labels)


class DnaDatasetID(Dataset):
    """
    DNA Dataset Dataset
    """
    def __init__(self, input_file, tokenizer, max_len, randomization=False):
        self.tokenizer = tokenizer
        self.input_file= input_file
        self.max_len = max_len
        self.randomization = randomization
        self._load_data_from_file()

    def _load_data_from_file(self):
        print('Loading data from file: {0}'.format(self.input_file))
        self.data = []
        with open(self.input_file) as fin:
            for line in fin:
                act_data = [int(token_id) for token_id in line.strip().split()][0:self.max_len]
                self.data.append(act_data)
        print('Doing randomization!')
        print(self.data)
        if self.randomization:
            random.shuffle(self.data)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # 
        return torch.tensor(self.data[index], dtype=torch.long)
    

class DnaDatasetHDF(Dataset):
    """ Ez azt csinálja, hogy a HDF5 file tartalmát beolvassa 
    """
    def __init__(self, input_file, tokenizer, max_len, randomization=False):
        self.tokenizer = tokenizer
        self.input_file= input_file
        self.max_len = max_len
        self.randomization = randomization
        self._load_data_from_file()

    def _load_data_from_file(self):
        print('Loading data from file: {0}'.format(self.input_file))
        dataset_file = h5py.File(self.input_file)
        ds_data = dataset_file['training_data']['sentences']
        self.data = torch.tensor(np.array(ds_data, dtype=np.int16), dtype=torch.long)
        dataset_file.close()

        if self.randomization:
            print('Doing randomization!')
            random.shuffle(self.data)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # 
        return self.data[index]
    



class FTDnaDatasetHDF(Dataset):
    """ HDF alapú fine tune dataset. Feltételezzük, hogy a hdf5 file tartalmazza a címkéket és az x-et. Valamint nincsen spec attention mask és egyéb gebasz. 
    """
    def __init__(self, input_file, tokenizer, sample_rate=1):
        self.tokenizer = tokenizer
        self.input_file= input_file
        self.sample_rate = sample_rate
        self._load_data_from_file()
        self.max_len = self.x.shape[1]
        self.permutation_tsh = 10000

    def _load_data_from_file(self):
        print('Loading data from file: {0}'.format(self.input_file))
        dataset_file = h5py.File(self.input_file)
        x = dataset_file['training_data']['x']
        y = dataset_file['training_data']['y']
        x = np.array(x, dtype=np.int16)
        y = np.array(y, dtype=np.int16)
        if self.sample_rate<1:
            sample_size = int(self.sample_rate*x.shape[0])
            rs = np.random.permutation(x.shape[0])
            print('Sampling the dataset! Sample size: ', sample_size)
            self.x_full = x
            self.y_full = y
            x = x[rs[0:sample_size],:]
            y = y[rs[0:sample_size]]
            print(x.shape)
            print(y.shape)
        self.x = torch.tensor(x, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)
        dataset_file.close()
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        item={'input_ids' : self.x[index,:]}
        item['labels'] = self.y[index]
        item['token_type_ids'] = torch.full(self.x[index,:].shape, 0, dtype=torch.long)
        item['attention_mask'] = torch.full(self.x[index,:].shape, 1, dtype=torch.long)
        if self.sample_rate<1 and index % self.permutation_tsh==0:
            print('Peforming resampling')
            sample_size = int(self.sample_rate*self.x_full.shape[0])
            rs = np.random.permutation(self.y_full.shape[0])
            self.x =  torch.tensor(self.x_full[rs[0:sample_size],:],dtype=torch.long)
            self.y =  torch.tensor(self.y_full[rs[0:sample_size]], dtype=torch.long)
            print(self.x.shape)
            print(self.y.shape)
        return item

class FTDnaItDatasetHDF(IterableDataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, input_batch_size=1000, randomized_batch=False, ds_offset=0, mode='pretrain'):
        self.tokenizer =tokenizer
        self.file_path = file_path
        self.input_batch_size = input_batch_size
        self.ds_offset=ds_offset
        self.mode = mode
        self.tokenized_file = file_path  # train.bash egyszerusitese miatt
        dataset_file = h5py.File(self.tokenized_file)
        self.dataset_sizes = list(dataset_file['training_data']['dataset_size'])
        self.ds_size= self.dataset_sizes[0]
        dataset_file.close()
        print('Dataset sizes: {0}'.format(self.dataset_sizes))

    def __len__(self):
        return self.dataset_sizes[0]

    def get_sentence(self, hdf_sentence):
        array_sent = np.array(hdf_sentence, dtype=np.int32)
        return torch.tensor(array_sent, dtype=torch.int32)

    def _get_fetch_interval(self):
        max_ds_items = self.dataset_sizes[0]
        #print(max_ds_items)
        new_fetch_start = self._act_ds_pointer * self.input_batch_size
        new_fetch_end = (self._act_ds_pointer + 1) * self.input_batch_size
        if new_fetch_end > max_ds_items:
            new_fetch_end = max_ds_items
        if new_fetch_start >= max_ds_items-1:
            self.dataset_file.close()
            raise StopIteration
        return new_fetch_start, new_fetch_end
    
    def __iter__(self):
        #print('Initiate an iterator!')


        self.dataset_file = h5py.File(self.tokenized_file)
        self.x = self.dataset_file['training_data']['x']
        self.y = self.dataset_file['training_data']['y']
        self._act_ds_pointer = math.floor(self.ds_offset/self.input_batch_size)+0
        print('act_starting ds pointer: ', self._act_ds_pointer)
        new_fetch_start, new_fetch_end = self._get_fetch_interval()
        print('Fetched intervals: ', new_fetch_start, new_fetch_end)

        self._x = np.array(self.x[new_fetch_start:new_fetch_end], dtype=np.int32)
        self._y = np.array(self.y[new_fetch_start:new_fetch_end], dtype=np.int32)
        self._act_ds_pointer_it = 0

        return self
    
    def __next__(self):
        ds_item = None

        if self._act_ds_pointer_it >= self.input_batch_size:
            self._act_ds_pointer_it = 0
            self._act_ds_pointer+=1
            try:
                new_fetch_start, new_fetch_end = self._get_fetch_interval()
            except StopIteration:
                print('Fethcing new data')
                self.dataset_file = h5py.File(self.tokenized_file)
                self.x = self.dataset_file['training_data']['x']
                self.y = self.dataset_file['training_data']['y']
                self._act_ds_pointer = 0
                self._act_ds_pointer_it = 0
                new_fetch_start, new_fetch_end = self._get_fetch_interval()
            self._x = np.array(self.x[new_fetch_start:new_fetch_end], dtype=np.int32)
            self._y = np.array(self.y[new_fetch_start:new_fetch_end], dtype=np.int32)
            try:
                ds_item={'input_ids' : self._x[self._act_ds_pointer_it,:]}
                ds_item['labels'] = self._y[self._act_ds_pointer_it]
                ds_item['token_type_ids'] = torch.full(self._x[self._act_ds_pointer_it,:].shape, 0, dtype=torch.uint8)
                ds_item['attention_mask'] = torch.full(self._x[self._act_ds_pointer_it,:].shape, 1, dtype=torch.uint8)
                self._act_ds_pointer_it+=1

            except IndexError:
                print('Index error')
                print(self._act_torchds, new_fetch_start, new_fetch_end)
        else:
            try:
                ds_item={'input_ids' : self._x[self._act_ds_pointer_it,:]}
                ds_item['labels'] = self._y[self._act_ds_pointer_it]
                ds_item['token_type_ids'] = torch.full(self._x[self._act_ds_pointer_it,:].shape, 0, dtype=torch.uint8)
                ds_item['attention_mask'] = torch.full(self._x[self._act_ds_pointer_it,:].shape, 1, dtype=torch.uint8)
                self._act_ds_pointer_it+=1
            except IndexError:
                print('IndexError!!!!')
                print('self._act_ds_pointer_it:', self._act_ds_pointer_it)
                print('self._act_ds_pointer', self._act_ds_pointer)
                print('Restart the iterations!')
                self.ds_offset=0
                self.dataset_file.close()
                self.dataset_file = h5py.File(self.tokenized_file)
                self.x = self.dataset_file['training_data']['x']
                self.y = self.dataset_file['training_data']['y']
                self._act_ds_pointer =  0
                print('act_starting ds pointer: ', self._act_ds_pointer)
                new_fetch_start, new_fetch_end = self._get_fetch_interval()
                print('Fetched intervals: ', new_fetch_start, new_fetch_end)

                self._x = np.array(self.x[new_fetch_start:new_fetch_end], dtype=np.int32)
                self._y = np.array(self.y[new_fetch_start:new_fetch_end], dtype=np.int32)
                self._act_ds_pointer_it = 0

                ds_item={'input_ids' : self._x[self._act_ds_pointer_it,:]}
                ds_item['labels'] = self._y[self._act_ds_pointer_it]
                ds_item['token_type_ids'] = torch.full(self._x[self._act_ds_pointer_it,:].shape, 0, dtype=torch.uint8)
                ds_item['attention_mask'] = torch.full(self._x[self._act_ds_pointer_it,:].shape, 1, dtype=torch.uint8)

                self._act_ds_pointer_it+=1



        #ds_item={'input_ids' : self.x[self._act_ds_pointer_it,:]}
        #ds_item['labels'] = self.y[self._act_ds_pointer_it]
        #ds_item['token_type_ids'] = torch.full(self.x[self._act_ds_pointer_it,:].shape, 0, dtype=torch.long)
        #ds_item['attention_mask'] = torch.full(self.x[self._act_ds_pointer_it,:].shape, 1, dtype=torch.long)
        #self._act_ds_pointer_it = self._act_ds_pointer_it + 1

        if self.mode == 'ft':
            #print(ds_item['input_ids'])
            ds_item['input_ids'] = torch.from_numpy(ds_item['input_ids']).long()
            ds_item['token_type_ids'] = ds_item['token_type_ids'].long()
            ds_item['attention_mask'] =ds_item['attention_mask'].long()

        ds_item['input_ids'][-2]=3
        ds_item['input_ids'][-1]=0
        return ds_item
    

class BabDataset(IterableDataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, input_batch_size=1000, randomized_batch=False, ds_offset=0):

        self.tokenizer =tokenizer
        self.file_path = file_path
        self.input_batch_size = input_batch_size
        self.randomized_batch =randomized_batch
        self.tokenized_file = file_path  # train.bash egyszerusitese miatt
        self.ds_offset = ds_offset # Azért, hogy a folytatáshoz ne kelljen végig iterálni a dataset-en. 400gb-nál ez elég sok idő :)
        dataset_file = h5py.File(self.tokenized_file)
        self.dataset_sizes = list(dataset_file['training_data']['dataset_size'])
        self.ds_size= self.dataset_sizes[0]
        dataset_file.close()
        print('Dataset sizes: {0}'.format(self.dataset_sizes))

    def __len__(self):
        return self.dataset_sizes[0]

    def get_sentence(self, hdf_sentence):
        array_sent = np.array(hdf_sentence, dtype=np.int16)
        return torch.tensor(array_sent, dtype=torch.int16)

    def _get_fetch_interval(self):

        max_ds_items = self.dataset_sizes[0]
        #print(max_ds_items)
        new_fetch_start = self._act_ds_pointer * self.input_batch_size
        new_fetch_end = (self._act_ds_pointer + 1) * self.input_batch_size
        if new_fetch_end > max_ds_items:
            new_fetch_end = max_ds_items
        if new_fetch_start >= max_ds_items-1:
            self.dataset_file.close()
            raise StopIteration
        return new_fetch_start, new_fetch_end



    def __next__(self):
        ds_item = None
        if self._act_ds_pointer_it >= self.input_batch_size:
            self._act_ds_pointer_it = 1
            self._act_ds_pointer+=1
            try:
                new_fetch_start, new_fetch_end = self._get_fetch_interval()
            except StopIteration:
                print('Fethcing new data')
                self.dataset_file = h5py.File(self.tokenized_file)
                self.dataset = self.dataset_file['training_data']['sentences']
                self._act_ds_pointer = 0
                self._act_ds_pointer_it = 0
                new_fetch_start, new_fetch_end = self._get_fetch_interval()
            self._act_torchds = np.array(self.dataset [new_fetch_start:new_fetch_end],
                                         dtype=np.int16)
            try:
                ds_item = self._act_torchds[0]
            except IndexError:
                print('Index error')
                print(self._act_torchds, new_fetch_start, new_fetch_end)

        else:
            try:
                ds_item = self._act_torchds[self._act_ds_pointer_it]
                self._act_ds_pointer_it+=1
            except IndexError:
                print('IndexError!!!!')
                print(self._act_torchds.shape)
                print('self._act_ds_pointer_it:', self._act_ds_pointer_it)
                print('self._act_ds_pointer', self._act_ds_pointer)
                print('Restart the iterations!')
                self.ds_offset=0
                self.dataset_file.close()
                self.dataset_file = h5py.File(self.tokenized_file)
                self.dataset = self.dataset_file['training_data']['sentences']
                self._act_ds_pointer = math.floor(self.ds_offset/self.input_batch_size)+0
                self._act_ds_pointer_it = 0
                new_fetch_start, new_fetch_end = self._get_fetch_interval()
                print('Fetched intervals: ', new_fetch_start, new_fetch_end)
                self._act_torchds = np.array(self.dataset [new_fetch_start:new_fetch_end], dtype=np.int16)
                ds_item = self._act_torchds[self._act_ds_pointer_it]
                self._act_ds_pointer_it+=1
                #raise IndexError
        return ds_item


    def __iter__(self):
        #print('Initiate an iterator!')

        self.dataset_file = h5py.File(self.tokenized_file)
        self.dataset = self.dataset_file['training_data']['sentences']
        self._act_ds_pointer = math.floor(self.ds_offset/self.input_batch_size)+0

        print('act_starting ds pointer: ', self._act_ds_pointer)

        self._act_ds_pointer_it = 0
        new_fetch_start, new_fetch_end = self._get_fetch_interval()
        print('Fetched intervals: ', new_fetch_start, new_fetch_end)
        self._act_torchds = np.array(self.dataset [new_fetch_start:new_fetch_end], dtype=np.int16)

        return self

# Ezt majd meg kellene írni TODO
#    def __getitem__(self, index):
#        # 
#        return torch.tensor(self.dataset_file['training_data']['sentences'][index], dtype=torch.int16)