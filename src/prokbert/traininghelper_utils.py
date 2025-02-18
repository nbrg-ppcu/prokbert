# This file defines the config classes that wil be used by all tests to hold
# meta information in a formatted way

import os
import json
import torch

from os import PathLike
from dataclasses import dataclass, asdict
from pathlib import Path
from string import ascii_letters, digits
from typing import Any, Union, Callable, Tuple
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import pandas as pd







def get_batch_size_komondor(basename: str, seq_len: int) -> Tuple[int, int, int]:
    if 'prokbert' in basename:
        return guess_initial_batch_size_komondor_prokbert(basemodel=basename, actL=seq_len)
    elif 'nucleotide' in basename:
        return guess_initial_batch_size_komondor_NT(basemodel=basename, seq_len=seq_len)
    elif 'DNABERT' in basename:
        return guess_initial_batch_size_komondor_DNABERT(basemodel=basename, seq_len=seq_len)

    raise RuntimeError('The basename {} does not match any known model'.format(basename))


def guess_initial_batch_size_komondor_prokbert(basemodel, actL):
    standard_params = {
        256: {'batch_size': 256, 'gradient_accumulation_steps': 1},
        512: {'batch_size': 128, 'gradient_accumulation_steps': 1},
        1024: {'batch_size': 32, 'gradient_accumulation_steps': 2},
    }

    long_params = {
        256: {'batch_size': 384, 'gradient_accumulation_steps': 1},
        512: {'batch_size': 256, 'gradient_accumulation_steps': 1},
        1024: {'batch_size': 128, 'gradient_accumulation_steps': 1},
        1536: {'batch_size': 64, 'gradient_accumulation_steps': 2},
    }

    standard_params = {
        256: {'batch_size': 512, 'gradient_accumulation_steps': 1},
        512: {'batch_size': 196, 'gradient_accumulation_steps': 1},
        1022: {'batch_size': 64, 'gradient_accumulation_steps': 2},
    }

    long_params = {
        256: {'batch_size': 512, 'gradient_accumulation_steps': 1},
        512: {'batch_size': 384, 'gradient_accumulation_steps': 1},
        1022: {'batch_size': 160, 'gradient_accumulation_steps': 1},
        1536: {'batch_size': 32, 'gradient_accumulation_steps': 2},
    }

    # Use long_params if model is a long variant, otherwise use standard_params
    if 'prokbert-mini-long' in basemodel:
        param_mapping = long_params
    else:
        param_mapping = standard_params

    # Ensure actL 1536 only for long model variants
    if actL == 1536 and 'prokbert-mini-long' not in basemodel:
        raise ValueError("Segment length 1536 is only valid for prokbert-mini-long.")

    # Sorted thresholds for parameter selection
    keys = sorted(param_mapping.keys())

    # Select the largest threshold that does not exceed actL
    if actL <= keys[0]:
        chosen_key = keys[0]
    else:
        chosen_key = None
        for k in keys:
            if k <= actL:
                chosen_key = k
            else:
                break

    if chosen_key is None:
        raise ValueError(f"Invalid segment length {actL} for the model {basemodel}.")

    batch_size = param_mapping[chosen_key]['batch_size']
    gradient_accumulation_steps = param_mapping[chosen_key]['gradient_accumulation_steps']
    return chosen_key, batch_size, gradient_accumulation_steps


def guess_initial_batch_size_komondor_DNABERT(basemodel, seq_len):
    param_mapping = {
        256: {'batch_size': 1024, 'gradient_accumulation_steps': 1},
        512: {'batch_size': 512, 'gradient_accumulation_steps': 2},
        1024: {'batch_size': 256, 'gradient_accumulation_steps': 1},
        2048: {'batch_size': 128, 'gradient_accumulation_steps': 2},
    }

    # Sorted thresholds for parameter selection
    keys = sorted(param_mapping.keys())

    # Select the largest threshold that does not exceed actL
    if seq_len <= keys[0]:
        chosen_key = keys[0]
    else:
        chosen_key = None
        for k in keys:
            if k <= seq_len:
                chosen_key = k
            else:
                break

    if chosen_key is None:
        raise ValueError(f"Invalid segment length {seq_len} for the model {basemodel}.")

    batch_size = param_mapping[chosen_key]['batch_size']
    gradient_accumulation_steps = param_mapping[chosen_key]['gradient_accumulation_steps']
    return chosen_key, batch_size, gradient_accumulation_steps


def guess_initial_batch_size_komondor_NT(basemodel, seq_len):
    small_params = {
        256: {'batch_size': 1024, 'gradient_accumulation_steps': 1},
        512: {'batch_size': 512, 'gradient_accumulation_steps': 2},
        1024: {'batch_size': 256, 'gradient_accumulation_steps': 1},
        2048: {'batch_size': 128, 'gradient_accumulation_steps': 2},
    }

    mid_params = {
        256: {'batch_size': 256, 'gradient_accumulation_steps': 1},
        512: {'batch_size': 128, 'gradient_accumulation_steps': 2},
        1024: {'batch_size': 64, 'gradient_accumulation_steps': 1},
        2048: {'batch_size': 32 , 'gradient_accumulation_steps': 2},
    }

    large_params = { #Effective BS 240
        256: {'batch_size': 96, 'gradient_accumulation_steps': 1},
        512: {'batch_size': 48, 'gradient_accumulation_steps': 1},
        1022: {'batch_size': 24, 'gradient_accumulation_steps': 2},
        2048: {'batch_size': 1, 'gradient_accumulation_steps': 1},
    }

    # Use long_params if model is a long variant, otherwise use standard_params
    if '50m' in basemodel:
        param_mapping = small_params
    elif '500m' in basemodel:
        param_mapping = mid_params
    elif '2.5b' in basemodel:
        param_mapping = large_params
    else:
        raise Exception('Unknown basemodel')

    # Sorted thresholds for parameter selection
    keys = sorted(param_mapping.keys())

    # Select the largest threshold that does not exceed actL
    if seq_len <= keys[0]:
        chosen_key = keys[0]
    else:
        chosen_key = None
        for k in keys:
            if k <= seq_len:
                chosen_key = k
            else:
                break

    if chosen_key is None:
        raise ValueError(f"Invalid segment length {seq_len} for the model {basemodel}.")

    batch_size = param_mapping[chosen_key]['batch_size']
    gradient_accumulation_steps = param_mapping[chosen_key]['gradient_accumulation_steps']
    return chosen_key, batch_size, gradient_accumulation_steps

########################################################################################################################
# Tokenize functions
# The NT and DNABERT func-s are exactly the same, but I wanted to maintain formula

def get_tokenize_function(model_name: str) -> Callable:
    if 'prokbert' in model_name:
        return tokenize_function_prokbert
    elif 'nucleotide' in model_name:
        return tokenize_function_NT
    elif 'DNABERT' in model_name:
        return tokenize_function_DNABERT
    else:
        raise ValueError(f"Unknown model name {model_name}.")

def tokenize_function_prokbert(examples, tokenizer):
    # Tokenize the input sequences
    encoded = tokenizer(
        examples["segment"],
        padding=True,
        add_special_tokens=True,
        return_tensors="pt",
    )

    # Get the input_ids and attention_mask
    input_ids = encoded["input_ids"].clone().detach()
    attention_mask = encoded["attention_mask"].clone().detach()

    # Mask tokens with IDs 2 and 3 in a vectorized way
    mask_tokens = (input_ids == 2) | (input_ids == 3)
    attention_mask[mask_tokens] = 0

    y = torch.tensor(examples["y"], dtype=torch.int64)

    # Return the updated dictionary
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": y,
    }


def tokenize_function_NT(examples, tokenizer, max_seq_len):
    # Tokenize the input sequences
    encoded = tokenizer(
        examples['segment'],
        padding='longest',
        add_special_tokens=True,
        truncation=True,
        return_tensors='pt',
        max_length=max_seq_len,
    )
    # Get the input_ids and attention_mask
    input_ids = encoded['input_ids'].clone().detach()
    attention_mask = encoded['attention_mask'].clone().detach()
    # Mask tokens with IDs 2 and 3 in a vectorized way
    mask_tokens = (input_ids == 2) | (input_ids == 3)
    attention_mask[mask_tokens] = 0
    y = torch.tensor(examples['y'], dtype=torch.int64)
    # Return the updated dictionary
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': y,
    }


def tokenize_function_DNABERT(examples, tokenizer, max_seq_len):
    # Tokenize the input sequences
    encoded = tokenizer(
        examples['segment'],
        padding='longest',
        add_special_tokens=True,
        truncation=True,
        return_tensors='pt',
        max_length=max_seq_len,
    )
    # Get the input_ids and attention_mask
    input_ids = encoded['input_ids'].clone().detach()
    attention_mask = encoded['attention_mask'].clone().detach()
    # Mask tokens with IDs 2 and 3 in a vectorized way
    mask_tokens = (input_ids == 2) | (input_ids == 3)
    attention_mask[mask_tokens] = 0
    y = torch.tensor(examples['y'], dtype=torch.int64)
    # Return the updated dictionary
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': y,
    }



@dataclass(init=False, order=True, eq=True)
class BaseHyperparameterConfig:

    def __init__(self, kwargs: Any):
        pass

    @staticmethod
    def _convert_path_to_string(value: Any) -> Any:
        if isinstance(value, (Path, os.PathLike)):
            return str(value)
        return value

    def to_json(self, path: Union[str, PathLike]) -> None:
        """Save config to JSON file."""
        if isinstance(path, str):
            path = Path(path)
        config_dict = {k.lstrip('_'): self._convert_path_to_string(v)
                       for k, v in asdict(self).items()}
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def from_json(cls, path: PathLike) -> 'BaseHyperparameterConfig':
        raise NotImplementedError('The base class should not be instantiated')





@dataclass(init=False, order=True, eq=True)
class TrainingHelperM(BaseHyperparameterConfig):
    _dataset_name: str
    _huggingface_prefix: str
    _basemodel: str
    _batch_size: int
    _epochs: float
    _gradient_accumulation_steps: int
    _learning_rate: float
    _separator: str            # Unique character combination to separate information in the finetuned name
    _seq_len: int
    _task: str ='phage'

    def __init__(self,
                 huggingface_model_name: str,
                 batch_size: int = None,
                 dataset_name: str = 'TEST',
                 epochs: float = 1.0,
                 gradient_accumulation_steps: int = None,
                 learning_rate: float = 0.001,
                 separator: str = '___',
                 seq_len: int = 512,
                 task: str = 'phage') -> None:

        assert 0 < epochs, "Please provide a valid epoch number. Got{}".format(epochs)
        assert 0 < learning_rate < 1, "Please provide a valid learning rate in [0 1] Got{}".format(learning_rate)

        self._huggingface_prefix, self._basemodel = huggingface_model_name.split('/')
        self._dataset_name = dataset_name
        self._epochs = epochs
        self._learning_rate = learning_rate
        self.separator = separator
        self._seq_len = seq_len
        self._task = task

        # Try to auto infer batch size grad acc steps if they are not provided
        if batch_size is None :
            _, self._batch_size, gac = get_batch_size_komondor(basename=huggingface_model_name, seq_len=seq_len)
        else:
            self._batch_size = batch_size
        if gradient_accumulation_steps is None :
            _, _, self._gradient_accumulation_steps = get_batch_size_komondor(basename=huggingface_model_name, seq_len=seq_len)
        else:
            self._gradient_accumulation_steps = gradient_accumulation_steps

        super().__init__(self)

    @property
    def huggingface_model_name(self) -> str:
        return self._huggingface_prefix + '/' + self._basemodel

    @huggingface_model_name.setter
    def huggingface_model_name(self, new_name: str):
        assert '/' in new_name, "Please provide a full Hugging Face model name in the format: <developer>/<model_name>, got{}".format(new_name)
        self._huggingface_prefix, self._basemodel = new_name.split('/')

    @property
    def base_model_name(self) -> str:
        return self._basemodel

    @base_model_name.setter
    def base_model_name(self, new_base_model_name: str) -> None:
        assert '/' not in new_base_model_name, "/ is a special character, it cannot be part of a base model name!"
        self._basemodel = new_base_model_name

    @property
    def epochs(self) -> float:
        return self._epochs

    @epochs.setter
    def epochs(self, epochs: Union[int, float]) -> None:
        assert 0 < epochs, "Epochs must be positive, got {}".format(epochs)
        self._epochs = float(epochs)

    @property
    def huggingface_prefix(self):
        return self._huggingface_prefix

    @huggingface_prefix.setter
    def huggingface_prefix(self, new_prefix: str):
        assert '/' not in new_prefix, "/ is a special character, it cannot be part of the prefix!"
        self._huggingface_prefix = new_prefix

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate: float) -> None:
        assert 0 < learning_rate < 1, "Learning rate must be between 0 and 1, got {}".format(learning_rate)
        self._learning_rate = learning_rate

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int) -> None:
        assert 0 < batch_size, "Batch size must be positive, got {}".format(batch_size)
        self._batch_size = batch_size

    @property
    def gradient_accumulation_steps(self) -> int:
        return self._gradient_accumulation_steps

    @gradient_accumulation_steps.setter
    def gradient_accumulation_steps(self, gradient_accumulation_steps: int) -> None:
        assert 0 < gradient_accumulation_steps, "Gradient accumulation steps must be positive, got {}".format(gradient_accumulation_steps)
        self._gradient_accumulation_steps = gradient_accumulation_steps

    @property
    def separator(self) -> str:
        return self._separator

    @separator.setter
    def separator(self, new_separator: str) -> None:
        assert new_separator, "Please provide a valid separator!"  # Ensures it's not empty

        # Define allowed special characters (excluding letters, numbers, '/', '\')
        forbidden_chars = ascii_letters + digits + "/\\"

        if any(char in forbidden_chars for char in new_separator):
            raise ValueError(f"Invalid separator: '{new_separator}'. Only special characters allowed, except for '/' and '\\'!")

        self._separator = new_separator

    @property
    def seq_len(self) -> int:
        return self._seq_len

    @seq_len.setter
    def seq_len(self, seq_len: int) -> None:
        assert 0 < seq_len, "Sequence length must be positive, got {}".format(seq_len)
        self._seq_len = seq_len

    @property
    def task(self) -> str:
        return self._task

    @task.setter
    def task(self, new_task: str) -> None:
        assert len(new_task) > 0, 'Please provide a valid task name!'
        self._task = new_task

    @property
    def train_split_name(self) -> str:
        return str(self._seq_len) + 'bp__train'

    @property
    def eval_split_name(self) -> str:
        return str(self._seq_len) + 'bp__val'

    @property
    def test_split_name(self) -> str:
        return str(self._seq_len) + 'bp__test'

    @classmethod
    def from_json(cls, path: Union[str, PathLike]) -> 'TrainingHelperM':
        with open(path, 'r') as f:
            data = json.load(f)

        prefix = data.pop('huggingface_prefix')
        basename = data.pop('basemodel')
        data['huggingface_model_name'] = prefix + '/' + basename
        return cls(**data)

    def get_default_model(self):
        return AutoModelForSequenceClassification.from_pretrained(self._huggingface_prefix + '/' + self._basemodel,
                                                                  trust_remote_code=True)

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(self._huggingface_prefix + '/' + self._basemodel, trust_remote_code=True)

    def get_tokenizer_function(self) -> Callable:
        return get_tokenize_function(self._huggingface_prefix + '/' + self._basemodel)

    @classmethod
    def initialize_from_environment(cls) -> 'TrainingHelperM':
        model_name_path = os.getenv('MODEL_NAME')
        if model_name_path is None:
            raise ValueError("MODEL_NAME environment variable is required")

        dataset_name = os.getenv('DATASET_NAME')
        if dataset_name is None:
            raise ValueError("DATASET_NAME environment variable is required")

        lr_str = os.getenv('LEARNING_RATE')
        if lr_str is None:
            raise ValueError("LEARNING_RATE environment variable is required")
        lr_rate = float(lr_str)

        ls_str = os.getenv('LS')
        if ls_str is None:
            raise ValueError("LS environment variable is required")
        actL = int(ls_str)

        epochs_str = os.getenv('NUM_TRAIN_EPOCHS')
        if epochs_str is None:
            raise ValueError("NUM_TRAIN_EPOCHS environment variable is required")
        num_train_epochs = float(epochs_str)

        task = os.getenv('TASK')
        if task is None:
            raise ValueError("TASK environment variable is required")

        separator = os.getenv('SEPARATOR')
        if separator is None:
            separator = '___'

        batch_size = os.getenv('BATCH_SIZE')
        gradient_accumulation_steps = os.getenv('GRADIENT_ACCUMULATION_STEPS')

        if batch_size is None or gradient_accumulation_steps is None:
            _, batch_size, gradient_accumulation_steps = get_batch_size_komondor(basename=model_name_path,
                                                                                 seq_len=actL)
        else: # If they exist convert them to integers
            batch_size = int(batch_size)
            gradient_accumulation_steps = int(gradient_accumulation_steps)


        return cls(huggingface_model_name=model_name_path,
                   dataset_name=dataset_name,
                   epochs=num_train_epochs,
                   learning_rate=lr_rate,
                   seq_len=actL,
                   batch_size=batch_size,
                   gradient_accumulation_steps=gradient_accumulation_steps,
                   task=task,
                   separator=separator
                   )

    def get_some_default_training_arguments():
        pass

    @classmethod
    def initialize_from_finetuned_name(cls, name: str, separator: str = '___') -> 'TrainingHelperM':
        huggingface_prefixes = {
            'prokbert': 'neuralbioinfo',
            'nucleotide': 'InstaDeepAI',
            'DNABERT': 'zhihan1996'

        }

        namelist = name.split(separator)
        model_name = namelist[1]

        # Get distributor name for full HF name
        hf_prefix = None
        for key in huggingface_prefixes:
            if key in model_name:
                hf_prefix = huggingface_prefixes[key]
        assert hf_prefix is not None, "Please provide a known model: Prokbert, Nuclelotide Transformer or DNABERT-2"

        return TrainingHelperM(huggingface_model_name=''.join([hf_prefix, '/', model_name]),
                               dataset_name=namelist[0],
                               task=namelist[2],
                               seq_len=int(namelist[3]),
                               epochs=float(namelist[4]),
                               learning_rate=float(namelist[5]))

    def get_finetuned_model_name(self) -> str:



        return (self._dataset_name
                + self._separator
                + self._basemodel
                + self._separator
                + self._task
                + self._separator
                + 'sl'  # Prefix for seq_len
                + str(self._seq_len)
                + self._separator
                + 'ep'  # Prefix for epochs
                + str(self._epochs)
                + self._separator
                + 'lr'  # Prefix for learning rate
                + str(self._learning_rate))


########################################################################################################################
# DATASET HELPER #
########################################################################################################################
# This si deprecated/is subject to heavy change
@dataclass(init=False, order=True, eq=True)
class TrainingHelperD(BaseHyperparameterConfig):
    _dataset_path: PathLike           # Path to the top level of the whole dataset
    _dataset_name: str               # Split to use TEST or LSOUT
    _seq_len: int                     # Sequence length
    _task: str ='phage'               # Training task, only phage for now
    _separator: str = '___'           # Unique character combination to separate information in the finetuned name

    def __init__(self,
                  dataset_path: Union[str, PathLike],
                  dataset_name: str,
                  seq_len: int,
                  task: str = 'phage',
                  separator: str = '___',
                  check_dataset_exist: bool = True) -> None:

        super().__init__(self)
        raise NotImplementedError("This class is not yet implemented!")
        if isinstance(dataset_path, str):
            dataset_path = Path(dataset_path)
        self._dataset_path = dataset_path
        self._dataset_name = dataset_name
        self._seq_len = seq_len
        self._task = task
        self._separator = separator
        if check_dataset_exist:
            assert os.path.exists(self._dataset_path) and os.path.isdir(self._dataset_path), (
                'Dataset path {} does not exist. Please check the path and try again.'.format(self._dataset_path)
            )


    # Getter and setter methods for every field

    @property
    def dataset_path(self) -> PathLike:
        return self._dataset_path

    @dataset_path.setter
    def dataset_path(self, new_path: Union[str, PathLike]) -> None:
        assert os.path.exists(new_path), 'Dataset path {} does not exist!'.format(new_path)
        assert os.path.isdir(new_path), 'Dataset path {} must be a directory!'.format(new_path)
        self._dataset_path = new_path

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    @dataset_name.setter
    def dataset_name(self, new_name: str) -> None:
        assert new_name is not None, 'Dataset name cannot be None!'
        self._dataset_name = new_name

    @property
    def seq_len(self) -> int:
        return self._seq_len

    @seq_len.setter
    def seq_len(self, new_seq_len: int) -> None:
        assert new_seq_len > 0, 'Please provide a valid sequence length!'
        self._seq_len = new_seq_len

    @property
    def task(self) -> str:
        return self._task

    @task.setter
    def task(self, new_task: str) -> None:
        assert len(new_task) > 0, 'Please provide a valid task name!'
        self._task = new_task

    @property
    def separator(self) -> str:
        return self._separator


    @separator.setter
    def separator(self, new_separator: str) -> None:
        assert new_separator, "Please provide a valid separator!"  # Ensures it's not empty

        # Define allowed special characters (excluding letters, numbers, '/', '\')
        forbidden_chars = ascii_letters + digits + "/\\"

        if any(char in forbidden_chars for char in new_separator):
            raise ValueError(f"Invalid separator: '{new_separator}'. Only special characters allowed, except for '/' and '\\'!")

        self._separator = new_separator

    @property
    def train_dataset_path(self) -> PathLike:
        return  Path(self._dataset_path) / (str(self.seq_len) + 'bp__train')

    @property
    def val_dataset_path(self) -> PathLike:
        return Path(self._dataset_path) / (str(self.seq_len) + 'bp__val')

    @property
    def test_dataset_path(self) -> PathLike:
        return Path(self._dataset_path) / (str(self.seq_len) + 'bp__test')

    @property
    def train_split_name(self) -> str:
        return str(self.seq_len) + 'bp__train'

    @property
    def eval_split_name(self) -> str:
        return str(self.seq_len) + 'bp__val'

    @property
    def test_split_name(self) -> str:
        return str(self.seq_len) + 'bp__test'


    @classmethod
    def from_json(cls, path: Union[str, PathLike], check_dataset_exist: bool = False) -> 'TrainingHelperD':
        with open(path, 'r') as f:
            data = json.load(f)

        data['dataset_path'] = Path(data['dataset_path'])

        return cls(**data, check_dataset_exist=check_dataset_exist)

    @classmethod
    def initialize_from_environment(cls, check_dataset_exist: bool = True) -> 'TrainingHelperD':
        dataset_name = os.getenv('DATASET_NAME')
        if dataset_name is None:
            raise ValueError("DATASET_NAME environment variable is required")
        dataset_path = os.getenv('DATASET_PATH')
        if dataset_path is None:
            raise ValueError("DATASET_PATH environment variable is required")
        seq_len = int(os.getenv('LS'))
        if seq_len is None:
            raise ValueError("LS (sequence length) environment variable is required")
        task = os.getenv('TASK')
        if task is None:
            raise ValueError("TASK environment variable is required")

        return cls(dataset_path=dataset_path,
                   dataset_name=dataset_name,
                   seq_len=seq_len,
                   task=task,
                   check_dataset_exist=check_dataset_exist)




class TrainingHelper():
    
    training_paramters = ['learning_rate', 
                          'batch_size', 
                          'gradient_accumulation_steps',
                          'max_token_length']
    
    parameter_group_mappings = {'sl': 'Ls',
                                'ep': 'epochs',
                                'lr': 'learning_rate',
                                'bs': 'batch_size',
                                'gac': 'gradient_accumulation_steps',
                                'mtl': 'max_token_length'}
    group_mappings_to_params = {TrainingHelper.parameter_group_mappings[k] : k for k in parameter_group_mappings.keys()}
    paramter_group_sep='___'



    def __init__(self):

        print('Init a training helper :) ')
        print('Reading the model database')

        self.load_model_database_from_google_spreadsheet()
        self.load_finetuning_helper_database()

        self.basemodels = set(self.model_db['hf_name'])



    
    def load_model_database_from_google_spreadsheet(self):

        print('Loading a google spreadsheet')

        sheet_id = '0'
        gid = '0'
        csv_url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}'

        sheet_id = '1uFNC-IS9MPfdsJSB9psOW5WM1_jhzwBljjJf51uZX8o'
        gid = '0' 
        csv_url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}'
        model_db = pd.read_csv(csv_url)        
        self.model_db = model_db
        

    def load_finetuning_helper_database(self):
        print('Loading the training helper utils with the default parameters')

        sheet_id = '1uFNC-IS9MPfdsJSB9psOW5WM1_jhzwBljjJf51uZX8o'
        gid = '752340417'

        # Construct the CSV export URL
        csv_url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}'

        # Read the CSV data into a pandas DataFrame
        finetuning_default_params = pd.read_csv(csv_url)

        self.finetuning_default_params = finetuning_default_params


    def get_my_finetunig_model_name(self, prefix, dataset, learning_rate=None, epochs=None, gradie Ls=None, batch_size=None):
        pass
        """ TEST___nucleotide-transformer-v2-50m-multi-species___phage___sl_256___ep_0.1___lr_5e-05
            prefix + short name + dataset + Ls + ep + learningrate + batchsize + gradient_accumulation_steps
        """
        
        


    def get_my_training_parameters(self, model, actLs=512, task_type = 'sequence_classification'):        
        "Query finetuning paramters"

        print(f'Getting finetuning parameters for the training for the model: {model}')
        print(f'Query Ls: {actLs}')

        if model not in self.basemodels:
            print(f'The model {model} is not in the database!')
            print(f'Supported models are: {self.basemodels}')
        
        data_answer = self.finetuning_default_params[ (self.finetuning_default_params['basemodel'] == model) &
                                                     (self.finetuning_default_params['seq_length_min'] < actLs) &
                                                     (self.finetuning_default_params['seq_length_max'] >= actLs)]
        
        print(data_answer)

        params = data_answer[self.training_paramters].to_dict()
        return params['basemodel']


        pass 
        