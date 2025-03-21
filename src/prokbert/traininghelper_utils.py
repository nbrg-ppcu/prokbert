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
import re
import pandas as pd




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



class TrainingHelper:
    """
    Helper class to provide model training information.
    
    It loads two databases:
      - A model database (basemodels) 
      - A training defaults database (finetuning_default_params)
    
    It also provides methods to query training parameters and to construct
    a standardized finetuning model name.
    """
    
    # List of training parameters (keys in the defaults CSV)
    training_parameters = [
        'learning_rate', 
        'batch_size', 
        'gradient_accumulation_steps',
        'max_token_length'
    ]
    
    # Mapping from abbreviated group to full parameter names
    parameter_group_mappings = {
        'ls': 'Ls',
        'ep': 'epochs',
        'lr': 'learning_rate',
        'bs': 'batch_size',
        'gac': 'gradient_accumulation_steps',
        'mtl': 'max_token_length'
    }
    
    # Reverse mapping: from full parameter name to its abbreviation.
    group_mappings_to_params = {v: k for k, v in parameter_group_mappings.items()}
    
    # Separator used for constructing model name strings.
    parameter_group_sep = '___'
    
    def __init__(self, excel_path: str = None):
        """
        Initialize the helper by loading model and training defaults databases.
        
        If `excel_path` is provided, the helper will try to load the sheets
        'basemodels' and 'defaultrtainingParameters' from that XLSX file.
        Otherwise, it defaults to loading from Google Sheets via CSV URLs.
        """
        print('Initializing TrainingHelper...')
        if excel_path is not None:
            self.load_from_excel(excel_path)
        else:
            self.load_model_database_from_google_spreadsheet()
            self.load_finetuning_helper_database()
            
        # Create a set of available base model names.
        self.basemodels = set(self.model_db['hf_name'])
    
    def load_from_excel(self, excel_path: str):
        """
        Load databases from an Excel file with sheets 'basemodels'
        and 'defaultrtainingParameters'.
        """
        print(f'Loading databases from Excel file: {excel_path}')
        try:
            self.model_db = pd.read_excel(excel_path, sheet_name='Basemodels')
            self.finetuning_default_params = pd.read_excel(excel_path, sheet_name='DefaultTrainingParameters')
            print('Successfully loaded Excel sheets.')
        except Exception as e:
            print(f'Error loading Excel file: {e}')
            raise
    
    def load_model_database_from_google_spreadsheet(self):
        """
        Load the model database from a Google Spreadsheet exported as CSV.
        """
        print('Loading model database from Google Spreadsheet...')
        # Replace these with the appropriate sheet id and gid.
        sheet_id = '1uFNC-IS9MPfdsJSB9psOW5WM1_jhzwBljjJf51uZX8o'
        gid = '0'
        csv_url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}'
        try:
            self.model_db = pd.read_csv(csv_url)
            print('Model database loaded.')
        except Exception as e:
            print(f'Error loading model database from Google Spreadsheet: {e}')
            raise
    
    def load_finetuning_helper_database(self):
        """
        Load the finetuning default parameters from a Google Spreadsheet exported as CSV.
        """
        print('Loading finetuning default parameters from Google Spreadsheet...')
        sheet_id = '1uFNC-IS9MPfdsJSB9psOW5WM1_jhzwBljjJf51uZX8o'
        gid = '752340417'
        csv_url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}'
        try:
            self.finetuning_default_params = pd.read_csv(csv_url)
            print('Finetuning defaults loaded.')
        except Exception as e:
            print(f'Error loading finetuning parameters: {e}')
            raise

    def get_my_finetunig_model_name(self, prefix: str, short_name: str, dataset: str,
                                    learning_rate=None, epochs=None,
                                    gradient_accumulation_steps=None, Ls=None,
                                    batch_size=None) -> str:
        """
        Construct a standardized finetuning model name.
        
        The name is composed as:
            prefix + short_name + dataset + each provided training parameter
        Each parameter is appended in the format: <abbr>_<value>,
        where the abbreviation is defined in parameter_group_mappings.
        
        Example:
            get_my_finetunig_model_name("TEST", "nucleotide-transformer-v2-50m-multi-species", "phage",
                                        learning_rate="5e-05", epochs="0.1", Ls=256)
            returns:
            "TEST___nucleotide-transformer-v2-50m-multi-species___phage___sl_256___ep_0.1___lr_5e-05"
        """
        parts = [prefix, short_name, dataset]
        
        # Build a dictionary of parameter values provided.
        params = {
            'Ls': Ls,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'gradient_accumulation_steps': gradient_accumulation_steps
        }
        
        # Append only those parameters that are not None.
        for param_name, value in params.items():
            if value is not None:
                # Look up the abbreviation; if not found, use the parameter name.
                abbr = self.group_mappings_to_params.get(param_name, param_name)
                parts.append(f"{abbr}{value}")
        
        # Join all parts with the designated separator.
        model_name = self.parameter_group_sep.join(parts)
        return model_name
    
    def get_my_training_parameters(self, model: str, actLs=512, task_type: str = 'sequence_classification'):
        """
        Query the finetuning default parameters for a given base model and token length.
        
        It checks that the model exists in the model database and then filters
        the defaults based on the token length (actLs).
        
        Returns:
            A dictionary of training parameters if found, with the following keys:
            - learning_rate (as-is, typically a float or string)
            - batch_size (int)
            - gradient_accumulation_steps (int)
            - max_token_length (int)
            
        Raises:
            ValueError: if the model is not in the database or no matching parameters are found.
        """
        print(f'Getting finetuning parameters for model: {model} with actLs: {actLs}')
        if model not in self.basemodels:
            error_msg = (f"The model {model} is not in the database! "
                        f"Supported models are: {self.basemodels}")
            print(error_msg)
            raise ValueError(error_msg)
        
        # Filter rows where the base model matches and actLs falls within the provided range.
        data_answer = self.finetuning_default_params[
            (self.finetuning_default_params['basemodel'] == model) &
            (self.finetuning_default_params['seq_length_min'] < actLs) &
            (self.finetuning_default_params['seq_length_max'] >= actLs)
        ]
        
        if data_answer.empty:
            error_msg = f"No training parameters found for model {model} with actLs {actLs}"
            print(error_msg)
            raise ValueError(error_msg)
        
        # Convert only the relevant parameters to a dictionary.
        params_dict = data_answer[self.training_parameters].iloc[0].to_dict()

        # Convert batch_size, gradient_accumulation_steps, and max_token_length to int.
        for key in ['batch_size', 'gradient_accumulation_steps', 'max_token_length']:
            if key in params_dict and params_dict[key] is not None:
                try:
                    params_dict[key] = int(params_dict[key])
                except ValueError:
                    raise ValueError(f"Value for {key} cannot be converted to an integer: {params_dict[key]}")
        
        return params_dict
    
    def parse_model_name(self, model_name: str) -> dict:
        """
        Parse a model folder name that uses the parameter_group_sep ("___") as separator.
        
        Expected format:
            prefix___short_name___dataset___<param_abbr><optional underscore><value>___...
            
        For example, the model name:
            'LeGO___prokbert-mini___phage___ls1024___ep4.0___lr0.0004___bs384___gac1'
        will be parsed into:
            {
                'prefix': 'LeGO',
                'short_name': 'prokbert-mini',
                'dataset': 'phage',
                    'Ls': 1024,
                    'epochs': 4.0,
                    'learning_rate': 0.0004,
                    'batch_size': 384,
                    'gradient_accumulation_steps': 1
                }
            }
        
        Note:
            Not all parameters have to be present in the name.
        """
        parts = model_name.split(self.parameter_group_sep)
        if len(parts) < 3:
            raise ValueError("Model name must contain at least prefix, short name, and dataset.")
        
        result = {
            "prefix": parts[0],
            "base_model": parts[1],
            "task": parts[2],
        }
        
        # Dynamically build the regex pattern using the keys from parameter_group_mappings.
        param_keys = list(self.parameter_group_mappings.keys())
        pattern_keys = "|".join(re.escape(key) for key in param_keys)
        pattern = re.compile(
            r'^(?P<abbr>(' + pattern_keys + r'))_?(?P<value>[-+]?\d*\.?\d+(?:e[-+]?\d+)?)$', 
            re.IGNORECASE
        )
        
        for part in parts[3:]:
            match = pattern.match(part)
            if match:
                abbr = match.group("abbr").lower()  # Normalize abbreviation to lower-case
                value_str = match.group("value")
                # Convert to float if it contains a decimal point or an exponent; otherwise, to int.
                if ('.' in value_str) or ('e' in value_str.lower()):
                    value = float(value_str)
                else:
                    value = int(value_str)
                
                # Map abbreviation to the full parameter name.
                full_param = self.parameter_group_mappings.get(abbr, abbr)
                result[full_param] = value
            else:
                print(f"Warning: Part '{part}' did not match expected parameter format.")
        
        return result



    def register_all_models(self, models_path: str) -> pd.DataFrame:
        """
        Recursively register all models and their checkpoints from the provided models_path.
        
        For each model directory (which follows the naming convention), the method parses
        the metadata using `parse_model_name`. Then, it finds checkpoint directories, extracts
        the checkpoint number, and gathers all information into a final DataFrame.
        
        Exception handling is added to continue processing even if one model or checkpoint fails.
        
        Returns:
            A pandas DataFrame with columns for the model metadata, checkpoint number,
            checkpoint path, and the original model directory name.
        """
        if not os.path.exists(models_path) or not os.listdir(models_path):
            raise ValueError(f"The provided models_path '{models_path}' does not exist or is empty.")
    
        # List to collect all registration records.
        records = []
    
        # List all directories inside models_path.
        model_dirs = [d for d in os.listdir(models_path) if os.path.isdir(os.path.join(models_path, d))]
        print(f"Found model directories: {model_dirs}")
    
        for model_dir in model_dirs:
            model_dir_path = os.path.join(models_path, model_dir)
            try:
                # Parse the model name to extract metadata.
                metadata = self.parse_model_name(model_dir)
            except Exception as e:
                print(f"Error parsing model directory '{model_dir}': {e}")
                continue  # Skip this model if parsing fails.
    
            # Find checkpoint directories in the current model directory.
            checkpoint_dirs = [d for d in os.listdir(model_dir_path)
                               if os.path.isdir(os.path.join(model_dir_path, d)) and "checkpoint-" in d]
    
            if not checkpoint_dirs:
                print(f"No checkpoint directories found for model '{model_dir}'.")
                continue
    
            for checkpoint_dir in checkpoint_dirs:
                try:
                    # Extract checkpoint number from the directory name.
                    cp_match = re.search(r"checkpoint-(\d+)", checkpoint_dir)
                    if not cp_match:
                        print(f"Could not parse checkpoint number in '{checkpoint_dir}' for model '{model_dir}'.")
                        continue
                    cp = int(cp_match.group(1))
    
                    checkpoint_path = os.path.join(model_dir_path, checkpoint_dir)
    
                    # Combine metadata with checkpoint information.
                    record = metadata.copy()
                    record["checkpoint"] = cp
                    record["checkpoint_path"] = checkpoint_path
                    record["model_directory"] = model_dir
    
                    records.append(record)
                except Exception as e:
                    print(f"Error processing checkpoint '{checkpoint_dir}' in model '{model_dir}': {e}")
                    continue
    
        # Convert the collected records into a DataFrame.
        df = pd.DataFrame(records)
        return df
    
    def select_preferred_checkpoints(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Given a DataFrame (as produced by register_all_models) that contains checkpoint information for each model,
        this method selects for each model the preferred checkpoint as follows:
          - If a checkpoint numbered 0 is available, that row is selected.
          - Otherwise, the row with the largest checkpoint number is selected.
        
        Returns:
            A pandas DataFrame with one row per model (identified by 'model_directory') corresponding to the preferred checkpoint.
        """
        preferred_records = []
        # Group by model_directory (i.e. the original model folder name)
        for model_dir, group in df.groupby("model_directory"):
            # If checkpoint 0 exists, choose that; otherwise, choose the row with the maximum checkpoint.
            if (group["checkpoint"] == 0).any():
                selected = group[group["checkpoint"] == 0].iloc[0]
            else:
                selected = group.loc[group["checkpoint"].idxmax()]
            preferred_records.append(selected)
        
        return pd.DataFrame(preferred_records)
    
    def get_tokenizer_for_basemodel(self, basemodel: str):
        """
        Given a base model name, return its tokenizer using the Hugging Face 'hf_path'
        from the model database. Remote code downloads are allowed.
        
        Args:
            basemodel (str): The base model name to lookup (should match the 'hf_name' column).
        
        Returns:
            AutoTokenizer: The loaded tokenizer for the specified base model.
        
        Raises:
            ValueError: If the basemodel is not found in the model database.
            Exception: If any error occurs during tokenizer loading.
        """
        # Check if the basemodel exists in the model database.
        if basemodel not in self.basemodels:
            raise ValueError(f"Basemodel '{basemodel}' not found in the model database. "
                             f"Available basemodels: {self.basemodels}")
        
        try:
            # Retrieve the row corresponding to the basemodel.
            model_row = self.model_db[self.model_db['hf_name'] == basemodel].iloc[0]
            hf_path = model_row['hf_path']
            
            # Load the tokenizer using the hf_path and allow remote code.
            tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)
            return tokenizer
        except Exception as e:
            print(f"Error loading tokenizer for basemodel '{basemodel}': {e}")
            raise