# Config utils
import yaml
import pathlib
from os.path import join
import os
import numpy as np
import torch
import argparse
from multiprocessing import cpu_count
from transformers import TrainingArguments
from copy import deepcopy
import re
import sys

class BaseConfig:
    """Base class for managing and validating configurations."""

    numpy_dtype_mapping = {1: np.int8,
                           2: np.int16,
                           8: np.int64,
                           4: np.int32}

    def __init__(self):
        super().__init__()

    def cast_to_expected_type(self, parameter_class: str, parameter_name: str, value: any) -> any:
        """
        Cast the given value to the expected type.

        :param parameter_class: The class/category of the parameter.
        :type parameter_class: str
        :param parameter_name: The name of the parameter.
        :type parameter_name: str
        :param value: The value to be casted.
        :type value: any
        :return: Value casted to the expected type.
        :rtype: any
        :raises ValueError: If casting fails.
        """
        expected_type = self.parameters[parameter_class][parameter_name]['type']

        if expected_type in ["integer", "int"]:
            try:
                return int(value)
            except ValueError:
                raise ValueError(f"Failed to cast value '{value}' to integer for parameter '{parameter_name}' in class '{parameter_class}'.")
        elif expected_type == "float":
            try:
                return float(value)
            except ValueError:
                raise ValueError(f"Failed to cast value '{value}' to float for parameter '{parameter_name}' in class '{parameter_class}'.")
        elif expected_type in ["string", "str"]:
            return str(value)
        elif expected_type in ["boolean", "bool"]:
            if isinstance(value, bool):
                return value
            elif str(value).lower() == "true":
                return True
            elif str(value).lower() == "false":
                return False
            else:
                raise ValueError(f"Failed to cast value '{value}' to boolean for parameter '{parameter_name}' in class '{parameter_class}'.")
        elif expected_type == "type":
            # For this type, we will simply return the value without casting. 
            # It assumes the configuration provides valid Python types.
            return value
        elif expected_type == "list":
            if isinstance(value, list):
                return value
            else:
                raise ValueError(f"Failed to validate value '{value}' as a list for parameter '{parameter_name}' in class '{parameter_class}'.")
        elif expected_type == "tuple":
            if isinstance(value, tuple):
                return value
            else:
                raise ValueError(f"Failed to validate value '{value}' as a tuple for parameter '{parameter_name}' in class '{parameter_class}'.")
        elif expected_type == "set":
            if isinstance(value, set):
                return value
            else:
                raise ValueError(f"Failed to validate value '{value}' as a set for parameter '{parameter_name}' in class '{parameter_class}'.")
        elif expected_type == "dict":
            if isinstance(value, dict):
                return value
            else:
                raise ValueError(f"Failed to validate value '{value}' as a dict for parameter '{parameter_name}' in class '{parameter_class}'.")
        else:
            raise ValueError(f"Unknown expected type '{expected_type}' for parameter '{parameter_name}' in class '{parameter_class}'.")



    def get_parameter(self, parameter_class: str, parameter_name: str) -> any:
        """
        Retrieve the default value of a specified parameter.

        :param parameter_class: The class/category of the parameter (e.g., 'segmentation').
        :type parameter_class: str
        :param parameter_name: The name of the parameter.
        :type parameter_name: str
        :return: Default value of the parameter, casted to the expected type.
        :rtype: any
        """
        default_value = self.parameters[parameter_class][parameter_name]['default']
        return self.cast_to_expected_type(parameter_class, parameter_name, default_value)
    

    
    def validate_type(self, parameter_class: str, parameter_name: str, value: any) -> bool:
        """
        Validate the type of a given value against the expected type.

        :param parameter_class: The class/category of the parameter.
        :type parameter_class: str
        :param parameter_name: The name of the parameter.
        :type parameter_name: str
        :param value: The value to be validated.
        :type value: any
        :return: True if the value is of the expected type, otherwise False.
        :rtype: bool
        """
        expected_type = self.parameters[parameter_class][parameter_name]['type']

        if expected_type == "integer" and not isinstance(value, int):
            return False
        elif expected_type == "float" and not isinstance(value, float):
            return False
        elif expected_type == "string" and not isinstance(value, str):
            return False
        else:
            return True
    
    def validate_value(self, parameter_class: str, parameter_name: str, value: any) -> bool:
        """
        Validate the value of a parameter against its constraints.

        :param parameter_class: The class/category of the parameter.
        :type parameter_class: str
        :param parameter_name: The name of the parameter.
        :type parameter_name: str
        :param value: The value to be validated.
        :type value: any
        :return: True if the value meets the constraints, otherwise False.
        :rtype: bool
        """
        constraints = self.parameters[parameter_class][parameter_name].get('constraints', {})
        
        if 'options' in constraints and value not in constraints['options']:
            return False
        if 'min' in constraints and value < constraints['min']:
            return False
        if 'max' in constraints and value > constraints['max']:
            return False
        return True
    

    def validate(self, parameter_class: str, parameter_name: str, value: any):
        """
        Validate both the type and value of a parameter.

        :param parameter_class: The class/category of the parameter.
        :type parameter_class: str
        :param parameter_name: The name of the parameter.
        :type parameter_name: str
        :param value: The value to be validated.
        :type value: any
        :raises TypeError: If the value is not of the expected type.
        :raises ValueError: If the value does not meet the parameter's constraints.
        """
        if not self.validate_type(parameter_class, parameter_name, value):
            raise TypeError(f"Invalid type for {parameter_name} for parameter class '{parameter_class}'. Expected {self.parameters[parameter_class][parameter_name]['type']}.")
        
        if not self.validate_value(parameter_class, parameter_name, value):
            raise ValueError(f"Invalid value for {parameter_name}  for parameter class '{parameter_class}'. Constraints: {self.parameters[parameter_class][parameter_name].get('constraints', {})}.")

    def describe(self, parameter_class: str, parameter_name: str) -> str:
        """
        Retrieve the description of a parameter.

        :param parameter_class: The class/category of the parameter.
        :type parameter_class: str
        :param parameter_name: The name of the parameter.
        :type parameter_name: str
        :return: Description of the parameter.
        :rtype: str
        """
        return self.parameters[parameter_class][parameter_name]['description']

    @staticmethod
    def rename_non_unique_parameters(config: dict) -> tuple[dict, dict, dict]:
        """
        Rename parameters in the configuration to ensure uniqueness across different groups.

        This method identifies parameters with the same name across different groups and renames them
        by prefixing the group name. This is to prevent conflicts when parameters are used in a context
        where the group name is not specified.

        :param config: A dictionary where each key is a group name and each value is a dict
                       of parameters for that group.
        :type config: dict

        :return: A tuple containing:
                 - renamed_config: A dictionary with the same structure as the input, but with non-unique parameter
                   names renamed. The structure is {group_name: {param_name: param_info}}.
                 - cmd_argument2group_param: A dictionary mapping the new parameter names to their original group
                   and parameter name. The structure is {new_param_name: [group_name, original_param_name]}.
                 - group2param2cmdarg: A dictionary mapping each group to a dict that maps the original parameter
                   names to the new parameter names. The structure is {group_name: {original_param_name: new_param_name}}.
        :rtype: tuple[dict, dict, dict]
        """

        # Identify non-unique parameter names
        param_counts = {}
        for group_name, parameters in config.items():
            for param_name in parameters.keys():
                param_counts[param_name] = param_counts.get(param_name, 0) + 1

        non_unique_params = {param for param, count in param_counts.items() if count > 1}

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

    @staticmethod
    def create_parser(config: dict) -> argparse.ArgumentParser:
        """
        Create and configure an argparse parser based on the given configuration.

        This method sets up a command-line argument parser with arguments defined in the configuration. 
        Each top-level key in the configuration represents a group of related arguments.

        :param config: A dictionary where each key is a group name and each value is a dict
                       of parameters for that group. Each parameter's information should include 
                       its type, default value, and help description.
        :type config: dict

        :return: Configured argparse.ArgumentParser instance with arguments added as specified
                 in the configuration.
        :rtype: argparse.ArgumentParser

        :raises ValueError: If an unknown or unsupported type is specified for a parameter.
        """
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
        excluded_parameters = ['vocabmap', 'np_tokentype', 'pretraining_dataset_data', 'optim']


        for group_name, parameters in config.items():
            group = parser.add_argument_group(group_name)
            for param_name, param_info in parameters.items():
                param_type_str = param_info['type']
                description = param_info['description']
                escaped_description = re.sub(r"([^%])%", r"\1%%", description)
                if param_name in excluded_parameters:
                    continue
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
                kwargs = {
                    'type': param_type,
                    'default': param_info['default'],
                    'help': escaped_description
                }            # Add constraints if they exist
                """
                if 'constraints' in param_info:
                    constraints = param_info['constraints']
                    if 'min' in constraints:
                        kwargs['type'] = lambda x: eval(param_type_str)(x) if eval(param_type_str)(x) >= constraints['min'] else sys.exit(f"Value for {param_name} must be at least {constraints['min']}")
                    if 'max' in constraints:
                        kwargs['type'] = lambda x: eval(param_type_str)(x) if eval(param_type_str)(x) <= constraints['max'] else sys.exit(f"Value for {param_name} must be at most {constraints['max']}")
                    if 'options' in constraints:
                        kwargs['choices'] = constraints['options']
                """
                # Add argument to the group
                group.add_argument(f'--{param_name}', **kwargs)
        return parser



class SeqConfig(BaseConfig):
    """Class to manage and validate sequence processing configurations."""

    def __init__(self):
        super().__init__()
        self.default_seq_config_file = self._get_default_sequence_processing_config_file()
        with open(self.default_seq_config_file, 'r') as file:
            self.parameters = yaml.safe_load(file)

        # Some postprocessing steps
        self.parameters['tokenization']['shift']['constraints']['max'] = self.parameters['tokenization']['kmer']['default']-1
        # Ha valaki update-li a k-mer paramter-t, akkor triggerelni kellene, hogy mi legyen. 

        self.get_and_set_segmentation_parameters()
        self.get_and_set_tokenization_parameters()
        self.get_and_set_computational_parameters()

    def _get_default_sequence_processing_config_file(self) -> str:
        """
        Retrieve the default sequence processing configuration file.

        :return: Path to the configuration file.
        :rtype: str
        """
        current_path = pathlib.Path(__file__).parent
        prokbert_seq_config_file = join(current_path, 'configs', 'sequence_processing.yaml')
        self.current_path = current_path

        try:
            # Attempt to read the environment variable
            prokbert_seq_config_file = os.environ['SEQ_CONFIG_FILE']
        except KeyError:
            # Handle the case when the environment variable is not found
            pass
            # print("SEQ_CONFIG_FILE environment variable has not been set. Using default value: {0}".format(prokbert_seq_config_file))
        return prokbert_seq_config_file

    
    def get_and_set_segmentation_parameters(self, parameters: dict = {}) -> dict:
        """
        Retrieve and validate the provided parameters for segmentation.

        :param parameters: A dictionary of parameters to be validated.
        :type parameters: dict
        :return: A dictionary of validated segmentation parameters.
        :rtype: dict
        :raises ValueError: If an invalid segmentation parameter is provided.
        """
        segmentation_params = {k: self.get_parameter('segmentation', k) for k in self.parameters['segmentation']}

        for param, param_value in parameters.items():
            if param not in segmentation_params:
                raise ValueError(f"The provided {param} is an INVALID segmentation parameter! The valid parameters are: {list(segmentation_params.keys())}")
            self.validate('segmentation', param, param_value)
            segmentation_params[param] = param_value
        self.segmentation_params = segmentation_params


        return segmentation_params


    def get_and_set_tokenization_parameters(self, parameters: dict = {}) -> dict:
        # Updating the other parameters if necesseary, i.e. if k-mer has-been changed, then the shift is updated and we run a parameter check at the end

        tokenization_params = {k: self.get_parameter('tokenization', k) for k in self.parameters['tokenization']}
        for param, param_value in parameters.items():
            if param not in tokenization_params:
                raise ValueError(f"The provided {param} is an INVALID tokenization parameter! The valid parameters are: {list(tokenization_params.keys())}")
            self.validate('tokenization', param, param_value)
            tokenization_params[param] = param_value

        # Loading and check the vocab file. It is assumed that its ordered dictionary
        vocabfile=tokenization_params['vocabfile']
        act_kmer = tokenization_params['kmer']
        if vocabfile=='auto':
            vocabfile_path = join(self.current_path, 'data/prokbert_vocabs/', f'prokbert-base-dna{act_kmer}', 'vocab.txt')
            tokenization_params['vocabfile'] = vocabfile_path
        else:
            vocabfile_path = vocabfile
        with open(vocabfile_path) as vocabfile_in:
            vocabmap = {line.strip(): i for i, line in enumerate(vocabfile_in)}
        tokenization_params['vocabmap'] = vocabmap

        # Loading the vocab
        self.tokenization_params = tokenization_params
        return tokenization_params    

    def get_and_set_computational_parameters(self, parameters: dict = {}) -> dict:
        """ Reading and validating the computational paramters
        """

        computational_params = {k: self.get_parameter('computation', k) for k in self.parameters['computation']}
        core_count = cpu_count()

        if computational_params['cpu_cores_for_segmentation'] == -1:
            computational_params['cpu_cores_for_segmentation'] = core_count

        if computational_params['cpu_cores_for_tokenization'] == -1:
            computational_params['cpu_cores_for_tokenization'] = core_count

        

        for param, param_value in parameters.items():
            if param not in computational_params:
                raise ValueError(f"The provided {param} is an INVALID computation parameter! The valid parameters are: {list(computational_params.keys())}")
            self.validate('computation', param, param_value)
            computational_params[param] = param_value

        np_tokentype= SeqConfig.numpy_dtype_mapping[computational_params['numpy_token_integer_prec_byte']]
        computational_params['np_tokentype'] = np_tokentype
        self.computational_params = computational_params
        return computational_params


    def get_maximum_segment_length_from_token_count_from_params(self):
        """Calculating the maximum length of the segment from the token count """
        max_token_counts = self.tokenization_params['token_limit']
        shift = self.tokenization_params['shift']
        kmer = self.tokenization_params['kmer']
        return self.get_maximum_segment_length_from_token_count(max_token_counts, shift, kmer)

    def get_maximum_token_count_from_max_length_from_params(self):
        """Calculating the maximum length of the segment from the token count """


        max_segment_length = self.tokenization_params['max_segment_length']
        shift = self.tokenization_params['shift']
        kmer = self.tokenization_params['kmer']          
        max_token_count = self.get_maximum_token_count_from_max_length(max_segment_length, shift, kmer)

        return max_token_count
    
    def get_cmd_arg_parser(self) -> tuple[argparse.ArgumentParser, dict, dict]:
        """
        Create and return a command-line argument parser for ProkBERT configurations, along with mappings 
        between command-line arguments and configuration parameters.

        This method combines sequence configuration parameters with training configuration parameters 
        and sets up a command-line argument parser using these combined settings. It ensures that parameter
        names are unique across different groups by renaming any non-unique parameters.

        :return: A tuple containing:
                 - Configured argparse.ArgumentParser instance for handling ProkBERT configurations.
                 - A dictionary mapping new command-line arguments to their original group and parameter name.
                 - A dictionary mapping each group to a dict that maps the original parameter names 
                   to the new command-line argument names.
        :rtype: tuple[argparse.ArgumentParser, dict, dict]

        Note: The method assumes that the configuration parameters for training and sequence configuration
        are available within the class.
        """
        combined_params = deepcopy(self.parameters)
        combined_params['Sequence'] = {}
        combined_params['Sequence']['fasta_file_dir'] = {'default': 'None',
                                                         'description' : 'Directory where the input fasta file are located for the pretraining',
                                                         'type': 'string'}
        combined_params['Sequence']['out'] = {'default': 'pretrain.h5',
                                                         'description' : 'Output path',
                                                         'type': 'string'}


        combined_params, cmd_argument2group_param, group2param2cmdarg = BaseConfig.rename_non_unique_parameters(combined_params)
        
        parser = BaseConfig.create_parser(combined_params)
        return parser,cmd_argument2group_param, group2param2cmdarg
    

    @staticmethod
    def get_maximum_segment_length_from_token_count(max_token_counts, shift, kmer):
        """Calcuates how long sequence can be covered
        """

        max_segment_length = (max_token_counts-3)*shift + kmer
        return max_segment_length

    @staticmethod
    def get_maximum_token_count_from_max_length(max_segment_length, shift, kmer):
        """Calcuates how long sequence can be covered
        """
        max_token_count = int(np.ceil((max_segment_length - kmer)/shift+3))
        return max_token_count

class ProkBERTConfig(BaseConfig):
    """Class to manage and validate pretraining configurations."""

    torch_dtype_mapping = {1: torch.uint8,
                           2: torch.int16,
                           8: torch.int64,
                           4: torch.int32}

    def __init__(self):
        super().__init__()

        self.default_pretrain_config_file = self._get_default_pretrain_config_file()
        with open(self.default_pretrain_config_file, 'r') as file:
            self.parameters = yaml.safe_load(file)
            
        # Load and validate each parameter set
        self.data_collator_params = self.get_set_parameters('data_collator')
        self.model_params = self.get_set_parameters('model')
        self.dataset_params = self.get_set_parameters('dataset')
        self.pretraining_params = self.get_set_parameters('pretraining')
        self.finetuning_params = self.get_set_parameters('finetuning')
        # Getting the sequtils params as well

        self.def_seq_config = SeqConfig()
        self.segmentation_params = self.def_seq_config.get_and_set_segmentation_parameters(self.parameters['segmentation'])
        self.tokenization_params = self.def_seq_config.get_and_set_tokenization_parameters(self.parameters['tokenization'])
        self.computation_params = self.def_seq_config.get_and_set_computational_parameters(self.parameters['computation'])

        self.default_torchtype = ProkBERTConfig.torch_dtype_mapping[self.computation_params['numpy_token_integer_prec_byte']]

        hf_training_args = TrainingArguments("working_dir")
        self.hf_training_args_dict = hf_training_args.to_dict()


    def _get_default_pretrain_config_file(self) -> str:
        """
        Retrieve the default pretraining configuration file.

        :return: Path to the configuration file.
        :rtype: str
        """
        current_path = pathlib.Path(__file__).parent
        pretrain_config_file = join(current_path, 'configs', 'pretraining.yaml')

        try:
            # Attempt to read the environment variable
            pretrain_config_file = os.environ['PRETRAIN_CONFIG_FILE']
        except KeyError:
            # Handle the case when the environment variable is not found
            pass
            # print(f"PRETRAIN_CONFIG_FILE environment variable has not been set. Using default value: {pretrain_config_file}")
        return pretrain_config_file
    
    def get_set_parameters(self, parameter_class: str, parameters: dict = {}) -> dict:
        """
        Retrieve and validate the provided parameters for a given parameter class.

        :param parameter_class: The class/category of the parameter (e.g., 'data_collator').
        :type parameter_class: str
        :param parameters: A dictionary of parameters to be validated.
        :type parameters: dict
        :return: A dictionary of validated parameters.
        :rtype: dict
        :raises ValueError: If an invalid parameter is provided.
        """
        class_params = {k: self.get_parameter(parameter_class, k) for k in self.parameters[parameter_class]}


        # First validatiading the class parameters as well
        for param, param_value in class_params.items():

            self.validate(parameter_class, param, param_value)


        for param, param_value in parameters.items():
            if param not in class_params and (parameter_class!='pretraining'):
                raise ValueError(f"The provided {param} is an INVALID {parameter_class} parameter! The valid parameters are: {list(class_params.keys())}")
            else:
                if parameter_class == 'pretraining' or parameter_class == 'finetuning' :
                    if param in self.hf_training_args_dict or param in class_params:
                        if param in class_params:
                            self.validate(parameter_class, param, param_value)
                        class_params[param] = param_value
                    else:
                        raise ValueError(f"The provided {param} is an INVALID {parameter_class} parameter! In addition is not a valid training argument.")
                else:
                    self.validate(parameter_class, param, param_value)
                    class_params[param] = param_value

        return class_params
    
    def get_and_set_model_parameters(self, parameters: dict = {}) -> dict:
        """ Setting the model parameters """

        # Here we include the additional training arguments available for the trainer

        self.model_params = self.get_set_parameters('model', parameters)

        return self.model_params

    def get_and_set_dataset_parameters(self, parameters: dict = {}) -> dict:
        """ Setting the dataset parameters """

        self.dataset_params = self.get_set_parameters('dataset', parameters)

        return self.dataset_params

    def get_and_set_pretraining_parameters(self, parameters: dict = {}) -> dict:
        """ Setting the model parameters """
        self.pretraining_params = self.get_set_parameters('pretraining', parameters)

        return self.pretraining_params       
    
    
    def get_and_set_datacollator_parameters(self, parameters: dict = {}) -> dict:
        """ Setting the model parameters """
        self.data_collator_params = self.get_set_parameters('data_collator', parameters)
        return self.data_collator_params
    
    def get_and_set_segmentation_parameters(self, parameters: dict = {}) -> dict:
        self.segmentation_params = self.def_seq_config.get_and_set_segmentation_parameters(parameters)

        return self.segmentation_params 
    def get_and_set_tokenization_parameters(self, parameters: dict = {}) -> dict:
        self.tokenization_params = self.def_seq_config.get_and_set_tokenization_parameters(parameters)
        
        return self.tokenization_params 
    def get_and_set_computation_params(self, parameters: dict = {}) -> dict:
        self.computation_params = self.def_seq_config.get_and_set_computational_parameters(parameters)
        return self.computation_params    

    def get_and_set_finetuning_parameters(self, parameters: dict = {}) -> dict:
        """ Setting the finetuning parameters """

        # Here we include the additional training arguments available for the trainer

        self.finetuning_params = self.get_set_parameters('finetuning', parameters)

        return self.finetuning_params
    

    def get_cmd_arg_parser(self, keyset=[]) -> tuple[argparse.ArgumentParser, dict, dict]:
        """
        Create and return a command-line argument parser for ProkBERT configurations, along with mappings 
        between command-line arguments and configuration parameters.

        This method combines sequence configuration parameters with training configuration parameters 
        and sets up a command-line argument parser using these combined settings. It ensures that parameter
        names are unique across different groups by renaming any non-unique parameters.

        :return: A tuple containing:
                 - Configured argparse.ArgumentParser instance for handling ProkBERT configurations.
                 - A dictionary mapping new command-line arguments to their original group and parameter name.
                 - A dictionary mapping each group to a dict that maps the original parameter names 
                   to the new command-line argument names.
        :rtype: tuple[argparse.ArgumentParser, dict, dict]

        Note: The method assumes that the configuration parameters for training and sequence configuration
        are available within the class.
        """
        if len(keyset) ==0:
            trainin_conf_keysets = ['data_collator', 'model', 'dataset', 'pretraining', 'finetuning']
        else:
            trainin_conf_keysets = keyset

        seq_config = deepcopy(self.def_seq_config.parameters)
        default_other_config = deepcopy(self.parameters)
        combined_params = {}
        for k,v in seq_config.items():
            combined_params[k] = v
        for k in trainin_conf_keysets:
            combined_params[k] = default_other_config[k]

        combined_params, cmd_argument2group_param, group2param2cmdarg = BaseConfig.rename_non_unique_parameters(combined_params)
        parser = BaseConfig.create_parser(combined_params)

        return parser,cmd_argument2group_param, group2param2cmdarg


def get_user_provided_args(args, parser):
    """
    Extract arguments provided by the user from the parsed arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        parser (argparse.ArgumentParser): The argument parser instance.

    Returns:
        dict: A dictionary of user-provided arguments and their values.
    """
        
    user_provided_args = {}
    for action in parser._actions:
        arg_name = action.dest
        default_value = action.default
        user_value = getattr(args, arg_name, None)
        if user_value != default_value:
            user_provided_args[arg_name] = user_value

    return user_provided_args

            





