import yaml
import pathlib
from os.path import join
import os
import sys
import argparse
import re

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
from prokbert.config_utils import ProkBERTConfig, get_user_provided_args
from prokbert.training_utils import *


seed=851115

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)


def prepare_input_arguments():
    """
    Prepare and validate input arguments for ProkBERT pretraining.

    Parses command-line arguments and sets the configuration for the pretraining process.

    Returns:
        ProkBERTConfig: Configuration object for ProkBERT pretraining.
    """
    prokbert_config = ProkBERTConfig()
    parser, cmd_argument2group_param, group2param2cmdarg = prokbert_config.get_cmd_arg_parser()
    args = parser.parse_args()
    user_provided_args = get_user_provided_args(args, parser)
    input_args2check = list(set(user_provided_args.keys()) - {'help'})
    parameter_group_names = list(prokbert_config.parameters.keys())
    # Initialization of the input parameterset
    parameters = {k: {} for k in parameter_group_names}
    for provided_input_argument in input_args2check:
        #print(f'Setting: {provided_input_argument}')
        param_group, param_name = cmd_argument2group_param[provided_input_argument]
        #print(f'It belongs to group: {param_group}. Maps to the parameter: {param_name}')
        act_value = getattr(args, provided_input_argument)
        parameters[param_group][param_name]=act_value    
        prokbert_config = ProkBERTConfig()
    
    
    _ = prokbert_config.get_and_set_model_parameters(parameters['model'])
    _ = prokbert_config.get_and_set_dataset_parameters(parameters['dataset'])
    _ = prokbert_config.get_and_set_pretraining_parameters(parameters['pretraining'])
    _ = prokbert_config.get_and_set_tokenization_parameters(parameters['tokenization'])
    _ = prokbert_config.get_and_set_segmentation_parameters(parameters['segmentation'])
    _ = prokbert_config.get_and_set_computation_params(parameters['computation'])
    _ = prokbert_config.get_and_set_datacollator_parameters(parameters['data_collator'])
    prokbert_config.default_torchtype = torch.long
    #print(user_provided_args)

    return prokbert_config


def main(prokbert_config):
    """
    Main function to run the ProkBERT pretraining pipeline.

    Initializes tokenizer, data collator, dataset, and model, and then starts the pretraining process.

    Args:
        prokbert_config (ProkBERTConfig): Configuration object containing all necessary parameters for pretraining.
    """
    check_nvidia_gpu()
    #print(prokbert_config.model_params)
    tokenizer = get_training_tokenizer(prokbert_config)
    prokbert_dc = get_data_collator_for_overlapping_sequences(tokenizer, prokbert_config)
    hdf_file_exists, ds_size = check_hdf_dataset_file(prokbert_config)
    dataset_path = prokbert_config.dataset_params['dataset_path']
    #print(dataset_path)
    training_dataset = ProkBERTPretrainingHDFDataset(dataset_path)
    #print(training_dataset[0:10])
    #print(training_dataset[0:10].shape)

    model = get_pretrained_model(prokbert_config)
    run_pretraining(model,tokenizer, prokbert_dc, training_dataset, prokbert_config)


    #print(input_args)

if __name__ == "__main__":
    print(f'Parsing')

    prokbert_config = prepare_input_arguments()
    main(prokbert_config)


    #main(input_args)


