# Training utils i.e. loading stuff, investiageting stuffs, etc
from config_utils import *
from sequtils import *
from prokbert_tokenizer import ProkBERTTokenizer
from ProkBERTDataCollator import *
from general_utils import *
from prok_datasets import *


def get_training_tokenizer(prokbert_config):
    """ Loading a tokenizer
    The input is the multiple paramterset requeired by the trainings
    It returns with a prokBert tokenizer
    """
    tokenizer = ProkBERTTokenizer(tokenization_params=prokbert_config.tokenization_params, 
                              segmentation_params=prokbert_config.segmentation_params,
                              comp_params=prokbert_config.computation_params,
                              operation_space='sequence')
    


    return tokenizer


def get_data_collator_for_overlapping_sequences(tokenizer, prokbert_config):

    print('Loading the datacollator class!')
    prokbert_dc = ProkBERTDataCollator(tokenizer,
                                    mask_to_left=prokbert_config.data_collator_params['mask_to_left'], 
                                    mask_to_right=prokbert_config.data_collator_params['mask_to_right'],
                                    mlm_probability =   prokbert_config.data_collator_params['mlm_probability'],
                                    replace_prob =prokbert_config.data_collator_params['replace_prob'],
                                    random_prob = prokbert_config.data_collator_params['random_prob'])
    prokbert_dc.set_torch_token_dtype(prokbert_config.default_torchtype)
    print(str(prokbert_dc))

    return prokbert_dc


def check_model_existance_and_checkpoint(model_name, output_path):
    """ The hugginface models are organized into checkpoints: except the final model
    The final checkpoint corresponds to the 0 checkpoint. Thus the checkpoint 0 thats the latest checkpoint
    """

    model_path = join(output_path,model_name)
    print('model_path:', model_path)
    path_exists = pathlib.Path.exists(pathlib.Path(model_path))
    largest_checkpoint_dir = None
    largest_checkpoint = None
    chekcpoint_nr = None
    print(model_path)
    if path_exists:
        try:
            subfolders = [ f for f in os.scandir(model_path) if f.is_dir() ]
            subfolders = [sf for sf in subfolders if len(os.listdir(subfolders[0])) > 1]
            chekcpoint_nr = sorted([int(f.name[11:]) for f in subfolders if f.name.startswith('checkpoint-')])
            largest_checkpoint = chekcpoint_nr[-1]
            if 0 in chekcpoint_nr:
                print('The 0 is the largest checkpoint!')
                largest_checkpoint = 0

            largest_checkpoint_dir = join(model_path, 'checkpoint-' + str(largest_checkpoint))

        except IndexError:
            print('Something is wrong, set default valies')
            print(str(subfolders))
            path_exists =False
            largest_checkpoint_dir = None
            largest_checkpoint = None

    return path_exists, largest_checkpoint_dir, largest_checkpoint, chekcpoint_nr


def check_hdf_dataset_file(prokbert_config):

    hdf_file_path = prokbert_config.dataset_params['dataset_path']
    dataset_class = prokbert_config.dataset_params['dataset_class']


    if len(hdf_file_path) == 0:
        raise(ValueError(f'There is no provided dataset file!'))
    
    print('Checking whether the file exists or not!')
    hdf_file_exists = check_file_exists(hdf_file_path)
    if dataset_class== 'IterableProkBERTPretrainingDataset':
        print('Loading and creating a IterableProkBERTPretrainingDataset')

        ds = IterableProkBERTPretrainingDataset(hdf_file_path)
        ds_size = len(ds)
    elif dataset_class== 'ProkBERTPretrainingHDFDataset':
        print('Loading and creating a IterableProkBERTPretrainingDataset')
        ds = ProkBERTPretrainingHDFDataset(hdf_file_path)
        ds_size = len(ds)
    elif dataset_class== 'ProkBERTPretrainingDataset':
        print('Checking the input data ...')
        ds_size = len(prokbert_config.dataset_params['pretraining_dataset_data'])
        if ds_size == 0:
            raise(ValueError(f'The provided data is empty, plase check the provided input.'))
    else:
        raise(ValueError(f'The rquired class={dataset_class} in not available'))
    


    return hdf_file_exists, ds_size

def get_the_iteration_offset(nr_gpus, act_gradient_accumulation_steps,
                             act_batch_size, largest_checkpoint, ds_size):


    act_ds_offset = nr_gpus*act_gradient_accumulation_steps*act_batch_size*largest_checkpoint % ds_size

    return act_ds_offset








