# Config utils
import yaml
import pathlib
from os.path import join
import os
import numpy as np

class SeqConfig:
    """Class to manage and validate sequence processing configurations."""

    def __init__(self):
        """
        Initialize the configuration, loading default parameters from the YAML file.
        """
        self.default_seq_config_file = self._get_default_sequence_processing_config_file()
        with open(self.default_seq_config_file, 'r') as file:
            self.parameters = yaml.safe_load(file)
        # Some postprocessing steps
        self.parameters['tokenization']['shift']['constraints']['max'] = self.parameters['tokenization']['kmer']['default']-1
        # Ha valaki update-li a k-mer paramter-t, akkor triggerelni kellene, hogy mi legyen. 

        self.get_segmentation_parameters()
        self.get_and_set_tokenization_params()


        

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
            print("SEQ_CONFIG_FILE environment variable has not been set. Using default value: {0}".format(prokbert_seq_config_file))
        return prokbert_seq_config_file

    def get_parameter(self, parameter_class: str, parameter_name: str) -> any:
        """
        Retrieve the default value of a specified parameter.

        :param parameter_class: The class/category of the parameter (e.g., 'segmentation').
        :type parameter_class: str
        :param parameter_name: The name of the parameter.
        :type parameter_name: str
        :return: Default value of the parameter.
        :rtype: any
        """
        return self.parameters[parameter_class][parameter_name]['default']
    
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
    
    def get_segmentation_parameters(self, parameters: dict = {}) -> dict:
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


    def get_and_set_tokenization_params(self, parameters: dict = {}) -> dict:
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
            print(self.current_path)
            vocabfile_path = join(self.current_path, 'data/prokbert_vocabs/', f'prokbert-base-dna{act_kmer}', 'vocab.txt')
            tokenization_params['vocabfile'] = vocabfile_path
        with open(vocabfile_path) as vocabfile_in:
            vocabmap = {line.strip(): i for i, line in enumerate(vocabfile_in)}
        tokenization_params['vocabmap'] = vocabmap

        # Loading the vocab
        self.tokenization_params = tokenization_params
        return tokenization_params
    

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
