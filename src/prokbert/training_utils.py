# Training utils i.e. loading stuff, investiageting stuffs, etc
from .config_utils import *
from .sequtils import *
from .prokbert_tokenizer import ProkBERTTokenizer
from .ProkBERTDataCollator import *
from .general_utils import *
from .prok_datasets import *
from .config_utils import *

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_training_tokenizer(prokbert_config: ProkBERTConfig) -> ProkBERTTokenizer:
    """
    Load a tokenizer for ProkBERT training.
    
    This function initializes and returns a ProkBERT tokenizer suitable for tokenizing sequences during training.

    :param prokbert_config: Configuration parameters for ProkBERT, an instance of :class:`ProkBERTConfig`.
                            The parameters are typically read from `pretraining.yaml`.
    :return: An instance of :class:`ProkBERTTokenizer`.
    """

    tokenizer = ProkBERTTokenizer(tokenization_params=prokbert_config.tokenization_params, 
                              segmentation_params=prokbert_config.segmentation_params,
                              comp_params=prokbert_config.computation_params,
                              operation_space='sequence')
    


    return tokenizer


def get_data_collator_for_overlapping_sequences(tokenizer, prokbert_config):
    """
    Load a data collator for overlapping sequences.
    
    This function initializes and returns a ProkBERT data collator suitable for handling overlapping sequences
    during training.

    :param tokenizer: The tokenizer to be used for tokenizing sequences. It should be an instance of :class:`ProkBERTTokenizer`.
    :param prokbert_config: Configuration parameters for ProkBERT. This should be an instance of :class:`ProkBERTConfig` 
                            and the parameters are typically read from the `pretraining.yaml` via the :class:`ProkBERTConfig`.
    :return: An instance of the :class:`ProkBERTDataCollator`.
    """

    logging.info('Loading the datacollator class!')
    prokbert_dc = ProkBERTDataCollator(tokenizer,
                                    mask_to_left=prokbert_config.data_collator_params['mask_to_left'], 
                                    mask_to_right=prokbert_config.data_collator_params['mask_to_right'],
                                    mlm_probability =   prokbert_config.data_collator_params['mlm_probability'],
                                    replace_prob =prokbert_config.data_collator_params['replace_prob'],
                                    random_prob = prokbert_config.data_collator_params['random_prob'])
    prokbert_dc.set_torch_token_dtype(prokbert_config.default_torchtype)
    logging.info(str(prokbert_dc))

    return prokbert_dc


def check_model_existance_and_checkpoint(model_name: str, output_path: str) -> Tuple[bool, Optional[str], Optional[int], Optional[List[int]]]:
    """
    Check the existence of a model and determine the latest checkpoint.
    
    The Hugging Face models are organized into checkpoints, with the final model corresponding to the "checkpoint 0".
    This function verifies the existence of a model in the specified output path and determines the checkpoint number 
    that represents the latest model.

    :param model_name: The name of the model to check.
    :param output_path: The path where the model checkpoints are stored.
    :return: 
        - True if the model exists, otherwise False.
        - Path to the largest checkpoint directory.
        - The largest checkpoint number.
        - A list of available checkpoint numbers.
    """


    model_path = join(output_path,model_name)
    logging.info( 'model_path:  ' +  str(model_path))
    path_exists = pathlib.Path.exists(pathlib.Path(model_path))
    largest_checkpoint_dir = None
    largest_checkpoint = None
    chekcpoint_nr = None
    if path_exists:
        try:
            subfolders = [ f for f in os.scandir(model_path) if f.is_dir() ]
            subfolders = [sf for sf in subfolders if len(os.listdir(subfolders[0])) > 1]
            chekcpoint_nr = sorted([int(f.name[11:]) for f in subfolders if f.name.startswith('checkpoint-')])
            largest_checkpoint = chekcpoint_nr[-1]
            if 0 in chekcpoint_nr:
                logging.info('   The 0 is the largest checkpoint!')
                largest_checkpoint = 0

            largest_checkpoint_dir = join(model_path, 'checkpoint-' + str(largest_checkpoint))

        except IndexError:
            logging.info('   Something is wrong, set default valies')
            logging.info('   ' + str(subfolders))
            path_exists =False
            largest_checkpoint_dir = None
            largest_checkpoint = None

    return path_exists, largest_checkpoint_dir, largest_checkpoint, chekcpoint_nr


def check_hdf_dataset_file(prokbert_config):
    """
    Verify the validity of an HDF5 dataset file.
    
    This function checks whether a given file path points to a valid HDF5 dataset used in ProkBERT's training.

    :param hdf_file_path: Path to the HDF5 dataset file.
    :return: True if the file is a valid HDF5 dataset, False otherwise.
    """

    hdf_file_path = prokbert_config.dataset_params['dataset_path']
    dataset_class = prokbert_config.dataset_params['dataset_class']


    if len(hdf_file_path) == 0:
        raise(ValueError(f'There is no provided dataset file!'))
    
    logging.info('Checking whether the file exists or not!')
    hdf_file_exists = check_file_exists(hdf_file_path)
    if dataset_class== 'IterableProkBERTPretrainingDataset':
        logging.info('Loading and creating a IterableProkBERTPretrainingDataset')

        ds = IterableProkBERTPretrainingDataset(hdf_file_path)
        ds_size = len(ds)
    elif dataset_class== 'ProkBERTPretrainingHDFDataset':
        logging.info('Loading and creating a IterableProkBERTPretrainingDataset')
        ds = ProkBERTPretrainingHDFDataset(hdf_file_path)
        ds_size = len(ds)
    elif dataset_class== 'ProkBERTPretrainingDataset':
        logging.info('Checking the input data ...')
        ds_size = len(prokbert_config.dataset_params['pretraining_dataset_data'])
        if ds_size == 0:
            raise(ValueError(f'The provided data is empty, plase check the provided input.'))
    else:
        raise(ValueError(f'The rquired class={dataset_class} in not available'))
    


    return hdf_file_exists, ds_size

def get_the_iteration_offset(batch_size, training_steps, dataset_size, 
                             nr_gpus=1, radient_accumulation_steps=1):
    """
    Determine the iteration offset for ProkBERT training.
    
    This function calculates the iteration offset based on the training configuration, output path, and tokenizer.
    It ensures that training resumes correctly from the last checkpoint or starts anew if no checkpoint is found.

    :param prokbert_config: Configuration parameters for ProkBERT, an instance of :class:`ProkBERTConfig`.
                            The parameters are typically read from `pretraining.yaml`.
    :param output_path: The path where the model checkpoints and other training artifacts are stored.
    :param tokenizer: An instance of :class:`ProkBERTTokenizer` used for tokenization during training.
    :param logger: Logger instance to log messages during the process.
    :param kwargs: Additional keyword arguments.
    :return: The iteration offset, indicating where the training should start or resume.
    """


    act_ds_offset = nr_gpus*radient_accumulation_steps*batch_size*training_steps % dataset_size

    return act_ds_offset


def get_pretrained_model(prokbert_config):
    from transformers import MegatronBertConfig, MegatronBertForMaskedLM

    new_model_args = MegatronBertConfig(**prokbert_config.model_params)

    [m_exists, cp_dir, cp, cps] = check_model_existance_and_checkpoint(prokbert_config.model_params['model_outputpath'], 
                                     prokbert_config.model_params['model_name'])
    if m_exists:
        print(f'Loading the existing model from the chekcpoint folder: {cp_dir}')
        expected_model_dir = cp_dir
        model = MegatronBertForMaskedLM.from_pretrained(expected_model_dir)
    else:
        print('Investigating whether previous model is exists')
        [init_m_exists, init_m_cp_dir, init_m_cp, init_m_cps] = check_model_existance_and_checkpoint(prokbert_config.model_params['model_outputpath'], 
                                        prokbert_config.model_params['model_name'])
        if not init_m_exists:
            print(f'The expected model does not exist {0}')
            model = MegatronBertForMaskedLM(new_model_args)
            
        else:
            model = MegatronBertForMaskedLM.from_pretrained(init_m_cp)


    return model
    

def run_pretraining(model,tokenizer, data_collator,training_dataset, prokbert_config):
    from transformers import Trainer, TrainingArguments

    training_args = TrainingArguments(**prokbert_config.pretraining_params)
    is_resume_training = prokbert_config.model_params['ResumeTraining']
    [m_exists, cp_dir, cp, cps] = check_model_existance_and_checkpoint(prokbert_config.model_params['model_outputpath'], 
                                     prokbert_config.model_params['model_name'])


    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=training_dataset,
        tokenizer=tokenizer
        )
    
    if is_resume_training and m_exists: 
        trainer.train(resume_from_checkpoint = cp_dir)
    else:
        trainer.train()






