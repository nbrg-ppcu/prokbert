import yaml
import pathlib
from os.path import join
import os
import sys
import argparse
import re
from collections import ChainMap

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

from transformers import MegatronBertForMaskedLM
from prokbert.prokbert_tokenizer import ProkBERTTokenizer
from transformers import MegatronBertModel, MegatronBertConfig, MegatronBertForMaskedLM

import pkg_resources
import random
import numpy as np
import torch

from os.path import join
from prokbert.sequtils import *
from prokbert.config_utils import SeqConfig
from prokbert.training_utils import get_training_tokenizer, get_data_collator_for_overlapping_sequences
from prokbert.prok_datasets import ProkBERTPretrainingHDFDataset
# Creating the model from scratch
from prokbert.config_utils import ProkBERTConfig
from prokbert.training_utils import *


seed=851115

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)

### This is an example for the a pretarining
def main(input_args):
    print('Running the pretraining from scratch!')
    #DATASET

    model_dir = input_args['output_dir']
    out_dataset_path = input_args['dataset_path']
    model_name = input_args['model_name']

    # Define segmentation and tokenization parameters
    segmentation_params = {
        'max_length': 256,  # Split the sequence into segments of length L
        'min_length': 6,
        'type': 'random'
    }
    tokenization_parameters = {
        'kmer': 6,
        'shift': 1,
        'max_segment_length': 2003,
        'token_limit': 2000
    }

    data_collator_params = {'mask_to_left' : 2,
                            'mask_to_right': 2,
                            'mlm_probability' : 0.025,
                            'replace_prob' : 0.8,
                            'random_prob' : 0.05
                            }

    model_params = {'model_outputpath': model_dir,
                    'model_name' : model_name,
                    'resume_or_initiation_model_path' : model_dir, 
                    'ResumeTraining' : False}

    dataset_params = {'dataset_path' : out_dataset_path}

    pretraining_params = {'output_dir': f'{model_dir}',
                        'warmup_steps' : 100,
                        'save_steps' : 200,
                        'save_total_limit' : 10,
                        'learning_rate' : 0.0004,
                        'gradient_accumulation_steps' : 1,
                        'per_device_train_batch_size': 32,
                        'num_train_epochs': 4}
    computation_params= {}
    # Setup configuration
    def_seq_config = SeqConfig()
    segmentation_params = def_seq_config.get_and_set_segmentation_parameters(segmentation_params)
    tokenization_params = def_seq_config.get_and_set_tokenization_parameters(tokenization_parameters)

    prokbert_config = ProkBERTConfig()

    _ = prokbert_config.get_and_set_model_parameters(model_params)
    _ = prokbert_config.get_and_set_dataset_parameters(dataset_params)
    _ = prokbert_config.get_and_set_pretraining_parameters(pretraining_params)
    _ = prokbert_config.get_and_set_tokenization_parameters(tokenization_params)
    _ = prokbert_config.get_and_set_segmentation_parameters(segmentation_params)
    _ = prokbert_config.get_and_set_computation_params(computation_params)
    _ = prokbert_config.get_and_set_datacollator_parameters(data_collator_params)


    prokbert_config.default_torchtype = torch.long
    tokenizer = get_training_tokenizer(prokbert_config)
    prokbert_dc = get_data_collator_for_overlapping_sequences(tokenizer, prokbert_config)
    #prokbert_dc.set_parameters()
    hdf_file_exists, ds_size = check_hdf_dataset_file(prokbert_config)
    training_dataset = ProkBERTPretrainingHDFDataset(out_dataset_path)

    model = get_pretrained_model(prokbert_config)
    run_pretraining(model,tokenizer, prokbert_dc,training_dataset, prokbert_config)


def rename_non_unique_parameters(config):
    # Identify non-unique parameter names
    param_counts = {}
    for group_name, parameters in config.items():
        for param_name in parameters.keys():
            param_counts[param_name] = param_counts.get(param_name, 0) + 1

    non_unique_params = {param for param, count in param_counts.items() if count > 1}
    print('non_unique_params: ', non_unique_params)

    cmd_argument2group_param = {}
    group2param2cmdarg = {}
    for group_name, parameters in config.items():
        group2param2cmdarg[group_name]={}
        for param_name in parameters.keys():
            group2param2cmdarg[group_name][param_name] = param_name


    # Rename only the non-unique parameters
    renamed_config = {}
    for group_name, parameters in config.items():
        renamed_group = {}
        for param_name, param_info in parameters.items():        

            new_param_name = f"{group_name}_{param_name}" if param_name in non_unique_params else param_name
            cmd_argument2group_param[new_param_name] = [group_name, param_name]
            group2param2cmdarg[group_name][param_name]=new_param_name

            renamed_group[new_param_name] = param_info
        renamed_config[group_name] = renamed_group
    return renamed_config, cmd_argument2group_param, group2param2cmdarg


def create_parser(config):
    parser = argparse.ArgumentParser(description="Command-line parser for project settings")


    # Mapping of type strings to Python types
    type_mapping = {
        'integer': int,
        'int': int,
        'float': float,
        'string': str,
        'str': str, 
        'bool': bool,
        'boolean': bool,
        'list': list
        # Complex types like 'dict' and 'type' are intentionally excluded
    }

    # List of types to handle as strings
    handle_as_string = ['dict', 'type', 'list']
    excluded_parameters = ['vocabmap', 'np_tokentype']


    for group_name, parameters in config.items():
        group = parser.add_argument_group(group_name)

        for param_name, param_info in parameters.items():
            print(group_name)
            print(param_name)
            print(param_info)
            print('______')
            param_type_str = param_info['type']
            description = param_info['description']
            escaped_description = re.sub(r"([^%])%", r"\1%%", description)


            if param_name in excluded_parameters:
                continue

            #print('__________')
            #print('param_name: ', param_name)
            #print('param_info: ', param_info)

            if param_type_str in handle_as_string:
                # Handle these types as strings in argparse, conversion will be done later in the program
                param_type = str

            elif param_type_str not in type_mapping:
                raise ValueError(f"Unknown or unsupported type '{param_type_str}' for parameter '{param_name}'")
            else:
                param_type = type_mapping[param_type_str]

            #print(f'The current type is: {param_type}')
            default_param = param_info['default']
            description = param_info['description']
            #print(f'The current default is: {default_param}')
            #print(f'The current description is: {escaped_description}')


            kwargs = {
                'type': param_type,
                'default': param_info['default'],
                'help': escaped_description
            }            # Add constraints if they exist
            if 'constraints' in param_info:
                constraints = param_info['constraints']
                if 'min' in constraints:
                    kwargs['type'] = lambda x: eval(param_info['type'])(x) if eval(param_info['type'])(x) >= constraints['min'] else sys.exit(f"Value for {param_name} must be at least {constraints['min']}")
                if 'max' in constraints:
                    kwargs['type'] = lambda x: eval(param_info['type'])(x) if eval(param_info['type'])(x) <= constraints['max'] else sys.exit(f"Value for {param_name} must be at most {constraints['max']}")
                if 'options' in constraints:
                    kwargs['choices'] = constraints['options']

            # Add argument to the group
            group.add_argument(f'--{param_name}', **kwargs)
            #print('Done')
    #print('parser seems to be setted')
    return parser

from copy import deepcopy

def parsing_arguments_loading_env_variables():

    #parser2 = argparse.ArgumentParser(description="Script to demonstrating the pretraining usage")
    default_datasetpath = pkg_resources.resource_filename('prokbert','data/pretraining_sample.h5')
    default_outputdir = '/tmp'
    default_params_file = ''
    default_model_name = 'prokbert-test'

    dataset_path =  os.getenv('DATASETPATH', default_datasetpath)
    output_dir = os.getenv('OUTPUTDIR', default_outputdir)
    model_name = os.getenv('MODELNAME', default_model_name)
    prokbert_params =  os.getenv('PARAMSFILE', default_params_file)
    print(dataset_path)

    #parser2.add_argument("--output_dir", type=str, default="/tmp",
    #                    help="Output directory for training logs and saved models.")
    #parser2.add_argument("--hdf_dataset_path", type=str, default=f"{default_datasetpath}",
    #                    help="Output directory for training logs and saved models.")

    # Adding auto configuration based on the YAML file? 

    prokbert_config = ProkBERTConfig()
    
    seq_config = deepcopy(prokbert_config.def_seq_config.parameters)
    default_other_config = deepcopy(prokbert_config.parameters)
    trainin_conf_keysets = ['data_collator', 'model', 'dataset', 'pretraining']

    combined_params = {}
    for k,v in seq_config.items():
        combined_params[k] = v
    for k in trainin_conf_keysets:
        combined_params[k] = default_other_config[k]

    #print(seq_config.keys())
    print(default_other_config.keys())
    #print(default_other_config['tokenization'])

    combined_params, cmd_argument2group_param, group2param2cmdarg = rename_non_unique_parameters(combined_params)
    print(group2param2cmdarg)

    #new_params = {'test' : seq_config['segmentation']}
    #print(new_params)
    #parser = create_parser(combined_params)

    args = parser.parse_args()
    
    input_args =  {'parsed_args': args,
                'dataset_path': dataset_path,
                'output_dir': output_dir,
                'prokbert_params' : prokbert_params,
                'model_name': model_name}
    

    return input_args


def prepare_input_arguments():

    prokbert_config = ProkBERTConfig()
    parser, cmd_argument2group_param, group2param2cmdarg = prokbert_config.get_cmd_arg_parser()
    args = parser.parse_args()

    return args, cmd_argument2group_param, group2param2cmdarg


if __name__ == "__main__":
    print(f'Parsing')

    input_args, cmd_argument2group_param, group2param2cmdarg = prepare_input_arguments()


    #main(input_args)


