import yaml
import pathlib
from os.path import join
import os
import sys
import argparse
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


def parsing_arguments_loading_env_variables():

    parser = argparse.ArgumentParser(description="Script to demonstrating the pretraining usage")
    default_datasetpath = pkg_resources.resource_filename('prokbert','data/pretraining_sample.h5')
    default_outputdir = '/tmp'
    default_params_file = ''
    default_model_name = 'prokbert-test'

    dataset_path =  os.getenv('DATASETPATH', default_datasetpath)
    output_dir = os.getenv('OUTPUTDIR', default_outputdir)
    model_name = os.getenv('MODELNAME', default_model_name)
    prokbert_params =  os.getenv('PARAMSFILE', default_params_file)
    print(dataset_path)

    parser.add_argument("--output_dir", type=str, default="/tmp",
                        help="Output directory for training logs and saved models.")
    parser.add_argument("--hdf_dataset_path", type=str, default=f"{default_datasetpath}",
                        help="Output directory for training logs and saved models.")

    # Adding auto configuration based on the YAML file? 


    args = parser.parse_args()
    
    input_args =  {'parsed_args': args,
                'dataset_path': dataset_path,
                'output_dir': output_dir,
                'prokbert_params' : prokbert_params,
                'model_name': model_name}
    

    return input_args




if __name__ == "__main__":
    pass
    print(f'Parsing')
    input_args = parsing_arguments_loading_env_variables()

    main(input_args)


