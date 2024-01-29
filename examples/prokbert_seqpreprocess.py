"""
This script is part of the ProkBERT examples, designed for preprocessing biological sequence data in preparation for pretraining.
Sequence Processing Pipeline:
Segmentation: Divides sequences into smaller, manageable segments for detailed analysis. The approach is detailed in the Segmentation Notebook, which outlines the methods and algorithms used for efficient sequence segmentation.
Tokenization: Transforms segmented sequence data into a tokenized format suitable for input into machine learning models. The procedure follows the guidelines presented in the Tokenization Notebook, providing insights into the tokenization strategy and its application in bioinformatics.

"""
from os.path import join
import random
import numpy as np
import torch

from os.path import join
from prokbert.sequtils import *
from prokbert.config_utils import SeqConfig, get_user_provided_args

seed=851115

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)





def prepare_input_arguments():
    """
    Prepare and validate input arguments for the data preprocessing pipeline.

    This function initializes the command-line argument parser, parses the arguments, 
    and validates the presence of required non-optional parameters.

    Detailed descriptions of the segmentation and tokenization processes can be found at:
    - Segmentation: https://github.com/nbrg-ppcu/prokbert/blob/release/examples/Segmentation.ipynb
    - Tokenization: https://github.com/nbrg-ppcu/prokbert/blob/release/examples/Tokenization.ipynb

    Returns:
        tuple: A tuple containing the SeqConfig object and parsed arguments.
    
    Raises:
        ValueError: If a required parameter is missing.
    """
    def_seq_config = SeqConfig()
    parser, cmd_argument2group_param, group2param2cmdarg = def_seq_config.get_cmd_arg_parser()
    args = parser.parse_args()
    user_provided_args = get_user_provided_args(args, parser)
    input_args2check = list(set(user_provided_args.keys()) - {'help'})
    non_optional_params = ['fasta_file_dir', 'out']
    for non_optional in non_optional_params:
        if non_optional not in input_args2check:
            print('The {non_optional} is required for the data preprocessing pipeline!')
            raise ValueError(f"Missing required parameters: {non_optional}")
    
    parameter_group_names = list(def_seq_config.parameters.keys())
    parameter_group_names = ['segmentation', 'tokenization', 'computation']
    #print(parameter_group_names)
    seq_params = list(set(input_args2check) - set(non_optional_params) )

    parameters = {k: {} for k in parameter_group_names}
    for provided_input_argument in seq_params:
        #print(f'Setting: {provided_input_argument}')
        param_group, param_name = cmd_argument2group_param[provided_input_argument]
        #print(f'It belongs to group: {param_group}. Maps to the parameter: {param_name}')
        act_value = getattr(args, provided_input_argument)
        parameters[param_group][param_name]=act_value    

    def_seq_config = SeqConfig()
    _ = def_seq_config.get_and_set_segmentation_parameters(parameters['segmentation'])
    _ = def_seq_config.get_and_set_tokenization_parameters(parameters['tokenization'])
    _ = def_seq_config.get_and_set_computational_parameters(parameters['computation'])

    #print(def_seq_config.tokenization_params)

    return def_seq_config, args


def main(seq_config, args):
    """
    Main function to run the dataset preprocessing pipeline.

    This function handles the loading and processing of sequence data, 
    including segmentation and tokenization, as described in the tokenization 
    and segmentation notebooks.

    Notebook Links:
    - Segmentation: https://github.com/nbrg-ppcu/prokbert/blob/release/examples/Segmentation.ipynb
    - Tokenization: https://github.com/nbrg-ppcu/prokbert/blob/release/examples/Tokenization.ipynb

    Args:
        seq_config (SeqConfig): The configuration object containing sequence processing parameters.
        args (argparse.Namespace): Parsed command-line arguments.
    """

    input_fasta_dir = getattr(args, 'fasta_file_dir')
    output_file = getattr(args, 'out')

    print(f'{input_fasta_dir}, {output_file}')
    segmentation_params = seq_config.segmentation_params
    tokenization_params = seq_config.tokenization_params
    computational_params = seq_config.computational_params

    #print(tokenization_params)
    #print(segmentation_params)



    # Load and segment sequences
    input_fasta_files = [join(input_fasta_dir, file) for file in get_non_empty_files(input_fasta_dir)]
    sequences = load_contigs(input_fasta_files, IsAddHeader=True, adding_reverse_complement=True, AsDataFrame=True, to_uppercase=True, is_add_sequence_id=True)
    segment_db = segment_sequences(sequences, segmentation_params, AsDataFrame=True)

    # Tokenization
    tokenized = batch_tokenize_segments_with_ids(segment_db, tokenization_params)
    expected_max_token = max(len(arr) for arrays in tokenized.values() for arr in arrays)
    X, torchdb = get_rectangular_array_from_tokenized_dataset(tokenized, tokenization_params['shift'], expected_max_token)

    # Save to HDF file
    save_to_hdf(X, output_file, database=torchdb, compression=True)



if __name__ == "__main__":
    print(f'Parsing the input arguments')

    seq_config,args = prepare_input_arguments()
    main(seq_config, args)


