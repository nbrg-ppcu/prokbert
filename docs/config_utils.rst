===========================
Configuration Utils
===========================

---------------------------
BaseConfig
---------------------------
.. autosummary::
    .. toctree:: api

        :maxdepth: 1
        :titlesonly:

        prokbert.config_utils.BaseConfig
        prokbert.config_utils.BaseConfig.cast_to_expected_type
        prokbert.config_utils.BaseConfig.get_parameter
        prokbert.config_utils.BaseConfig.validate_type
        prokbert.config_utils.BaseConfig.validate_value
        prokbert.config_utils.BaseConfig.validate
        prokbert.config_utils.BaseConfig.describe
    

.. autoclass:: prokbert.config_utils.BaseConfig
    :members:

    
--------------------------
SeqConfig
--------------------------
.. autosummary::
    .. toctree:: api

        :maxdepth: 1
        :titlesonly:
        

        prokbert.config_utils.SeqConfig
        prokbert.config_utils.SeqConfig._get_default_sequence_processing_config_file
        prokbert.config_utils.SeqConfig.get_and_set_segmentation_parameters
        prokbert.config_utils.SeqConfig.get_and_set_tokenization_parameters
        prokbert.config_utils.SeqConfig.get_and_set_computational_parameters
        prokbert.config_utils.SeqConfig.get_maximum_segment_length_from_token_count_from_params
        prokbert.config_utils.SeqConfig.get_maximum_segment_length_from_token_count
        prokbert.config_utils.SeqConfig.get_maximum_token_count_from_max_length

.. autoclass:: prokbert.config_utils.SeqConfig
    :members:
    :show-inheritance:

--------------------------
ProkBERTConfig
--------------------------
.. autosummary::
    .. toctree:: api

        :maxdepth: 1
        :titlesonly:

        prokbert.config_utils.ProkBERTConfig
        prokbert.config_utils.ProkBERTConfig._get_default_pretrain_config_file
        prokbert.config_utils.ProkBERTConfig.get_set_parameters
        prokbert.config_utils.ProkBERTConfig.get_and_set_model_parameters
        prokbert.config_utils.ProkBERTConfig.get_and_set_dataset_parameters
        prokbert.config_utils.ProkBERTConfig.get_and_set_pretraining_parameters
        prokbert.config_utils.ProkBERTConfig.get_and_set_datacollator_parameters
        prokbert.config_utils.ProkBERTConfig.get_and_set_segmentation_parameters
        prokbert.config_utils.ProkBERTConfig.get_and_set_tokenization_parameters
        prokbert.config_utils.ProkBERTConfig.get_and_set_computation_params

.. autoclass:: prokbert.config_utils.ProkBERTConfig
    :members:
    :show-inheritance:


---------------------------
Config YAMLs
---------------------------


.. literalinclude:: ../src/prokbert/configs/pretraining.yaml
    :language: yaml
    :caption: pretraining.yaml

.. literalinclude:: ../src/prokbert/configs/sequence_processing.yaml
    :language: yaml
    :caption: sequence_processing.yaml
    :name: sequence_processing.yaml

