# Training utils i.e. loading stuff, investiageting stuffs, etc
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, matthews_corrcoef, confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from scipy.special import logit, expit  
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import Trainer,  get_linear_schedule_with_warmup, EvalPrediction
from importlib import import_module
from torch.optim import AdamW
from typing import List, Tuple, Dict, Union

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
            subfolders = [ f for f in os.scandir(model_path) if f.is_dir() and f.name.startswith('checkpoint-')]
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
    from transformers import MegatronBertConfig, MegatronBertForMaskedLM, BertForMaskedLM, BertConfig

    #new_model_args = MegatronBertConfig(**prokbert_config.model_params)
    #model = MegatronBertForMaskedLM(new_model_args)

    new_model_args = MegatronBertConfig(**prokbert_config.model_params)
    #model = BertForMaskedLM(new_model_args)
    #return model


    [m_exists, cp_dir, cp, cps] = check_model_existance_and_checkpoint(prokbert_config.model_params['model_outputpath'], 
                                     prokbert_config.model_params['model_name'])
    if m_exists:
        print(f'Loading the existing model from the chekcpoint folder: {cp_dir}')
        expected_model_dir = cp_dir
        model = MegatronBertForMaskedLM.from_pretrained(expected_model_dir)
    else:
        print('Investigating whether previous model is exists')
        [init_m_exists, init_m_cp_dir, init_m_cp, init_m_cps] = check_model_existance_and_checkpoint(prokbert_config.model_params['resume_or_initiation_model_path'], 
                                        prokbert_config.model_params['model_name'])
        if not init_m_exists:
            model_output_path = prokbert_config.model_params['model_outputpath']
            print(f'The expected model does not exist at the path {model_output_path}. Creating a new modell with parameters: {new_model_args}')
            model = MegatronBertForMaskedLM(new_model_args)
            
        else:
            model = MegatronBertForMaskedLM.from_pretrained(init_m_cp_dir)


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
        #trainer.train()
        trainer.train(resume_from_checkpoint = cp_dir)

    else:
        trainer.train()
    final_model_output = join(prokbert_config.model_params['model_outputpath'], prokbert_config.model_params['model_name'])
    model.save_pretrained(final_model_output)


def oevaluate_binary_classification_bert_build_pred_results(logits: torch.Tensor, labels: torch.Tensor) -> np.ndarray:
    """
    Build prediction results for binary classification.
    
    Parameters:
        logits (torch.Tensor): Raw model outputs for each class.
        labels (torch.Tensor): True labels.
        
    Returns:
        np.ndarray: An array containing labels, predictions, and logits for each class.
    """
    
    predictions = torch.argmax(logits, dim=-1)
    p = predictions.detach().cpu().numpy()
    y = labels.detach().cpu().numpy()
    logits = logits.detach().cpu().numpy()
    pred = np.stack((y, p)).T
    pred_results = np.concatenate((pred, logits), axis=1)
    
    return pred_results

def evaluate_binary_classification_bert_build_pred_results(logits, labels):
    """
    Build prediction results for binary classification.
    
    Parameters:
        logits: Raw model outputs (logits) as a tensor or numpy array.
        labels: True labels as a tensor or numpy array.
        
    Returns:
        np.ndarray: An array containing labels, predictions, and logits for each class.
    """
    
    # Ensure logits and labels are torch Tensors
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits)
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)
    
    # Calculate predictions
    predictions = torch.argmax(logits, dim=-1)
    
    # Move tensors to CPU and convert to numpy for concatenation
    predictions_np = predictions.cpu().numpy()
    labels_np = labels.cpu().numpy()
    logits_np = logits.cpu().numpy()
    
    # Prepare the results
    pred = np.stack((labels_np, predictions_np), axis=1)
    pred_results = np.concatenate((pred, logits_np), axis=1)
    
    return pred_results



def evaluate_binary_classification_bert(pred_results: np.ndarray) -> Tuple[Dict, List]:
    y_true = pred_results[:, 0]
    y_pred = pred_results[:, 1]
    logits = pred_results[:, 2:]  # Logits for both classes
    
    # Compute probabilities using the sigmoid function on the logits for the positive class (index 1)
    probabilities = expit(logits[:, 1])
    
    # Calculate Cross-Entropy Loss
    cross_entropy_loss = -np.mean(y_true * np.log(probabilities) + (1 - y_true) * np.log(1 - probabilities))
    
    try:
        auc_class1 = roc_auc_score(y_true, logits[:, 0])
    except ValueError:
        auc_class1 = -1
    
    try:
        auc_class2 = roc_auc_score(y_true, logits[:, 1])
    except ValueError:
        auc_class2 = -1
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred) 

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    Np = tp + fn
    Nn = tn + fp
    
    eval_results = {
        'CE_loss': cross_entropy_loss,
        'auc_class0': auc_class1,
        'auc_class1': auc_class2,
        'acc': acc,
        'bal_acc': bal_acc,
        'f1': f1,
        'mcc': mcc,
        'recall': recall,
        'sensitivity': recall,
        'specificity': specificity,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
        'Np': Np,
        'Nn': Nn
    }
    
    eval_results_ls = [cross_entropy_loss, auc_class1, auc_class2,acc,bal_acc,mcc, f1, tn, fp, fn, tp, Np, Nn]
    
    return eval_results, eval_results_ls



def oldevaluate_binary_classification_bert(pred_results: np.ndarray) -> Tuple[Dict, List]:
    """
    Calculate various metrics for binary classification based on the prediction results.
    
    Parameters:
        pred_results (np.ndarray): An array containing labels, predictions, and logits for each class.
        
    Returns:
        Tuple[Dict, List]:
            - Dict: A dictionary containing various evaluation metrics.
            - List: A list containing some of the metrics for further analysis.
    """
    
    y_true = pred_results[:, 0]
    y_pred = pred_results[:, 1]
    class_0_scores = pred_results[:, 2]
    class_1_scores = pred_results[:, 3]
    
    try:
        auc_class1 = roc_auc_score(y_true, class_0_scores)
    except ValueError:
        auc_class1 = -1
    
    try:
        auc_class2 = roc_auc_score(y_true, class_1_scores)
    except ValueError:
        auc_class2 = -1
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred) 

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    Np = tp + fn
    Nn = tn + fp
    
    eval_results = {
        'auc_class0': auc_class1,
        'auc_class1': auc_class2,
        'acc': acc,
        'bal_acc': bal_acc,
        'f1': f1,
        'mcc': mcc,
        'recall': recall,
        'sensitivity': recall,
        'specificity': specificity,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
        'Np': Np,
        'Nn': Nn
    }
    
    eval_results_ls = [auc_class1, auc_class2, f1, tn, fp, fn, tp, Np, Nn]    
    return eval_results, eval_results_ls




def compute_metrics_eval_prediction(eval_preds: EvalPrediction) -> Dict:
    eval_preds_tuple = eval_preds.predictions, eval_preds.label_ids
    eval_results = compute_metrics(eval_preds_tuple)

    return eval_results



def compute_metrics(eval_preds: Tuple) -> Dict:
    """
    Compute metrics for binary classification evaluation.
    
    Parameters:
        eval_preds (Tuple): A tuple containing two elements:
            - logits: A list or array of raw model outputs.
            - labels: A list or array of true labels.
            
    Returns:
        Dict: A dictionary containing evaluation metrics.
        
    Note:
        This function assumes that `evaluate_binary_classification_bert_build_pred_results`
        and `evaluate_binary_classification_bert` are available in the scope.
    """

    logits, labels = eval_preds
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)
    # Generate prediction results (assuming this function is available in the scope)
    pred_results = evaluate_binary_classification_bert_build_pred_results(logits, labels)
    # Evaluate binary classification (assuming this function is available in the scope)
    eval_results, eval_results_ls = evaluate_binary_classification_bert(pred_results)
   
    return eval_results


class ProkBERTTrainer(Trainer):
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        # Create AdamW optimizer with the largest learning rate
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,  # Largest learning rate
            eps=self.args.adam_epsilon,
            weight_decay=self.args.weight_decay,
        )
        
        #optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,  # Default to 0, but you can specify a different number
            num_training_steps=num_training_steps  # This is typically the number of epochs * number of batches per epoch
        )        
        self.optimizer = optimizer
        self.lr_scheduler = scheduler


def get_torch_data_from_segmentdb_classification(tokenizer, segmentdb, L=None, randomize=True):

    if L is None:
        L = tokenizer.tokenization_params['token_limit']-2

    tokenized_sets = batch_tokenize_segments_with_ids(segmentdb, tokenizer.tokenization_params, 
                                                    batch_size=50000,
                                                    num_cores=tokenizer.comp_params['cpu_cores_for_tokenization'], 
                                                    np_token_type= np.int32)
    
    X, torchdb = get_rectangular_array_from_tokenized_dataset(tokenized_sets, 
                                                shift=tokenizer.tokenization_params['shift'],
                                                max_token_count=L+2,
                                                randomize=randomize,
                                                truncate_zeros = True,
                                                numpy_dtype = np.int32)
    
    torchdb_annot = torchdb.merge(segmentdb[['segment_id', 'y', 'label']], how='left', left_on = 'segment_id', right_on = 'segment_id')
    y=torch.tensor(torchdb_annot['y'], dtype=torch.long)
    X = torch.tensor(X, dtype=torch.long)

    return X, y, torchdb

def get_default_pretrained_model_parameters(model_name: str, model_class: str, output_hidden_states: bool = False,
                                            output_attentions: bool = False, move_to_gpu: bool = True):
    """
    Load a default pretrained model along with the corresponding tokenizer based on the model name.
    
    :param model_name: The name of the model to load. Should be a valid model stored locally or registered in the database.
                       Can be provided with or without the 'neuralbioinfo/' prefix.
    :type model_name: str
    :param model_class: The class of the transformer model into which the parameters will be loaded.
    :type model_class: str
    :param output_hidden_states: Whether to output hidden states.
    :type output_hidden_states: bool
    :param output_attentions: Whether to output attentions.
    :type output_attentions: bool
    :param move_to_gpu: Whether to move the model to GPU if available.
    :type move_to_gpu: bool
    :return: The loaded model (moved to GPU or CPU as specified) and the tokenizer with its default parameters.
    :rtype: tuple
    
    Raises:
        ValueError: If the model name does not match the expected pattern and is not found in predefined exceptions.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Normalize the model name by removing the 'neuralbioinfo/' prefix if present
    normalized_model_name = model_name.replace('neuralbioinfo/', '')

    print(f'normalized_model_name: {normalized_model_name}, model name_ {model_name}')
    # Predefined exceptions for model names and their tokenization parameters
    model_tokenization_params = {
        'prokbert-mini': {'kmer': 6, 'shift': 1},
        'prokbert-mini-long': {'kmer': 6, 'shift': 2},
        'prokbert-mini-c': {'kmer': 1, 'shift': 1},
    }
    
    # Check for predefined exceptions first
    if normalized_model_name in model_tokenization_params:
        tokenization_params = model_tokenization_params[normalized_model_name]
    else:
        # If not found, try to parse using regex
        match = re.search(r'k(\d+)s(\d+)', normalized_model_name)
        if match:
            kmer, shift = map(int, match.groups())
            tokenization_params = {'kmer': kmer, 'shift': shift}
        else:
            print('fdsgfdgfgfggfgfgf')
            raise ValueError(f"Model name '{model_name}' does not match the expected pattern and is not a predefined exception.")
    
    tokenizer = ProkBERTTokenizer(tokenization_params=tokenization_params, operation_space='sequence')
    model = load_pretrained_model(
        model_path=model_name,  # Use original model_name here to preserve 'neuralbioinfo/' if it was included
        model_class=model_class,
        device=device,
        output_hidden_states=output_hidden_states,
        output_attentions=output_attentions,
        move_to_gpu=move_to_gpu
    )

    return model, tokenizer
    



def load_pretrained_model(model_path, model_class, device, output_hidden_states=False, output_attentions=False, move_to_gpu=False):
    """
    Load Megatron BERT model and prepare for evaluation.

    Parameters:
    model_path (str): Path to the model.
    device (str): Device to load the model onto.

    Returns:
    MegatronBertForMaskedLM: Loaded model.
    """
    torch.cuda.empty_cache()
    ModelClass = getattr(import_module('transformers'), model_class)
    model = ModelClass.from_pretrained(model_path, output_attentions=output_attentions,output_hidden_states=output_hidden_states)
    #model = torch.compile(model)

    if move_to_gpu:
        model.to(device)
        num_gpus = torch.cuda.device_count()
        print('num_gpus: ', num_gpus)
        print('No of parameters: ', model.num_parameters()/1000000)
        if num_gpus > 1 and torch.cuda.device_count() >= num_gpus:
            model = torch.nn.DataParallel(model, device_ids=list(range(num_gpus)))
    return model


def check_nvidia_gpu():
    """
    Check if NVIDIA GPU is available for PyTorch and print an appropriate message.
    """
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_names = ', '.join(torch.cuda.get_device_name(i) for i in range(gpu_count))
        print(f"NVIDIA GPU is available. Total GPUs: {gpu_count}, Names: {gpu_names}")
    else:
        print("NVIDIA GPU is not available.")


def check_amd_gpu():
    """
    Check if AMD GPU is available for PyTorch (ROCm) and print an appropriate message.
    """
    # This is a placeholder function. PyTorch does not natively support AMD GPUs as of now.
    # Checking for AMD GPU support requires specific setup and installation of PyTorch with ROCm.
    print("Checking for AMD GPU is not directly supported in PyTorch as of now.")
    print("For AMD GPU support, ensure PyTorch is installed with ROCm and consult the ROCm documentation.")

def weighted_voting(df):
    """
    Performs weighted voting based on probabilities for each segment within the same sequence.
    
    Parameters:
    - df: DataFrame expected to contain at least the following columns:
          'sequence_id' and probability columns.
          The probability columns can be named either as 'p_class_0' and 'p_class_1'
          or as 'p_class0' and 'p_class1'. They will be normalized to 'p_class_0' and 'p_class_1'.
          Optionally, the DataFrame may contain a 'y' column representing the true label.
    
    Returns:
    - DataFrame with columns ['sequence_id', 'y_true', 'y_pred', 'score_class_0', 'score_class_1'].
      If the input DataFrame does not contain a 'y' column, the 'y_true' column in the output will be NaN.
    """
    # Normalize probability column names if needed.
    rename_dict = {}
    if 'p_class0' in df.columns and 'p_class_0' not in df.columns:
        rename_dict['p_class0'] = 'p_class_0'
    if 'p_class1' in df.columns and 'p_class_1' not in df.columns:
        rename_dict['p_class1'] = 'p_class_1'
    if rename_dict:
        df = df.rename(columns=rename_dict)
    
    # Determine required columns based on whether 'y' is available.
    required_cols = ['sequence_id', 'p_class_0', 'p_class_1']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Step 1: Calculate mean probabilities for each class by sequence_id.
    mean_probs = df.groupby('sequence_id')[['p_class_0', 'p_class_1']].mean()
    
    # Step 2: Determine predicted class based on higher mean probability.
    mean_probs['y_pred'] = mean_probs.idxmax(axis=1).map({'p_class_0': 0, 'p_class_1': 1})
    
    # Step 3: Check if 'y' column is provided. If yes, join with the true labels.
    if 'y' in df.columns:
        # Assuming 'y' is consistent for all segments within the same sequence.
        y_true = df[['sequence_id', 'y']].drop_duplicates().set_index('sequence_id')
        result_df = mean_probs.join(y_true)
        # Rename 'y' column to 'y_true'.
        result_df.rename(columns={'y': 'y_true'}, inplace=True)
    else:
        # If no 'y' column, add a column of NaNs.
        result_df = mean_probs.copy()
        result_df['y_true'] = np.nan
    
    # Step 4: Rename probability columns to score columns and reorder.
    result_df.rename(columns={
        'p_class_0': 'score_class_0', 
        'p_class_1': 'score_class_1'
    }, inplace=True)
    
    result_df = result_df.reset_index()[['sequence_id', 'y_true', 'y_pred', 'score_class_0', 'score_class_1']]
    
    return result_df




def prevweighted_voting(df):
    """
    Performs weighted voting based on probabilities for each segment within the same sequence.
    
    Parameters:
    - df: DataFrame with columns ['sequence_id', 'y', 'p_class_0', 'p_class1']
    
    Returns:
    - DataFrame with columns ['sequence_id', 'y_true', 'y_pred', 'score_class_0', 'score_class_1']
    """
    # Step 1: Calculate mean probabilities for each class by sequence_id
    mean_probs = df.groupby('sequence_id')[['p_class_0', 'p_class1']].mean()
    
    # Step 2: Determine predicted class based on higher mean probability
    mean_probs['y_pred'] = mean_probs.idxmax(axis=1).map({'p_class_0': 0, 'p_class1': 1})
    
    # Step 3: Join with original labels to get y_true for each sequence_id
    # Assuming 'y' is the same for all segments within the same sequence
    y_true = df[['sequence_id', 'y']].drop_duplicates().set_index('sequence_id')
    result_df = mean_probs.join(y_true)
    
    # Step 4: Rename and reorder columns to match required output
    result_df.rename(columns={'y': 'y_true', 'p_class_0': 'score_class_0', 'p_class1': 'score_class_1'}, inplace=True)
    result_df = result_df.reset_index()[['sequence_id', 'y_true', 'y_pred', 'score_class_0', 'score_class_1']]
    
    return result_df



def logits_to_sequence_predictions(df):
    # Calculate the mean logits for each class by sequence_id
    mean_logits = df.groupby('sequence_id')[['logit_y0', 'logit_y1']].mean().reset_index()
    
    # Convert the mean logits to a numpy array for softmax computation
    logits_array = mean_logits[['logit_y0', 'logit_y1']].to_numpy()
    # Apply softmax to the mean logits to get probabilities
    probabilities = softmax(logits_array, axis=1)
    
    # Get the sequence-level label (y_true) by taking the first occurrence since it should be the same for all segments of a sequence
    y_true = df.groupby('sequence_id')['y'].first().reset_index()['y']
    
    # Determine sequence-level prediction (y_pred) based on the highest probability
    y_pred = np.argmax(probabilities, axis=1)
    
    # Create a results DataFrame
    results_df = pd.DataFrame({
        'sequence_id': mean_logits['sequence_id'],
        'y_true': y_true,
        'y_pred': y_pred,
        'score_class_0': probabilities[:, 0],
        'score_class_1': probabilities[:, 1]
    })
    
    return results_df


def evaluate_binary_sequence_predictions(predictions, segment_dataset):
    
    
    final_cols = ['sequence_id', 'predicted_label', 'p_class_0', 'p_class_1']

    logits = predictions.predictions
    labels = predictions.label_ids

    # Convert logits and labels to tensors
    logits_tensor = torch.tensor(logits)
    labels_tensor = torch.tensor(labels)

    print('Building predictions...')
    pred_results = evaluate_binary_classification_bert_build_pred_results(logits_tensor, labels_tensor)
    logits = pred_results[:, 2:]
    probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    combined_results = np.concatenate([pred_results, probabilities], axis=1)
    combined_results = pd.DataFrame(combined_results,
                                    columns=['y_true', 'y_pred', 'logit_y0', 'logit_y1', 'p_class_0', 'p_class_1'])

    segment_dataset_df = segment_dataset.select_columns(['sequence_id', 'y']).to_pandas()

    merged_results = segment_dataset_df.merge(combined_results, left_index=True, right_index=True)
    merged_results.rename({'p_class_1' : 'p_class1', 
                        'p_class_0' : 'p_class_0'}, axis=1, inplace=True)
    print('Applying weighted voting...')
    sequence_predictions = weighted_voting(merged_results)
    sequence_predictions.rename({'score_class_0': 'p_class_0', 'score_class_1' : 'p_class_1'}, axis=1, inplace=True)
    sequence_predictions['predicted_label'] = np.where(sequence_predictions['y_pred'] == 1, 'class_1', 'class_0')

    seq_pred_results =sequence_predictions[['y_true', 'y_pred', 'p_class_0', 'p_class_1']].to_numpy()
    probabilities = seq_pred_results[:, 2:]
    logits = np.log(probabilities)
    seq_pred_results[:,2:]=logits
    seq_eval_results, seq_eval_results_ls = evaluate_binary_classification_bert(seq_pred_results)
    #print('Sequence level performance: ')
    #print(seq_eval_results)
    #print('_________________')

    final_table = sequence_predictions[final_cols]

    #print('Evaluating the other results')
    #eval_results, eval_results_ls = evaluate_binary_classification_bert(pred_results)

    return final_table, seq_eval_results



def inference_binary_sequence_predictions(predictions, segment_dataset):
    """
    Inference-only version that aggregates binary classification predictions for sequences.
    
    Args:
        predictions: An object with an attribute 'predictions' containing the model logits.
                     (No true labels are expected.)
        segment_dataset: A dataset with a 'sequence_id' column (and no ground truth label column).
    
    Returns:
        final_table: A DataFrame with columns ['sequence_id', 'predicted_label', 'p_class_0', 'p_class_1']
                     that holds the aggregated, sequence-level predictions.
    """
    # Final output columns
    final_cols = ['sequence_id', 'predicted_label', 'p_class_0', 'p_class_1']

    # Extract logits (shape: [n_samples, 2]) from predictions.
    logits = predictions.predictions

    # Convert logits to tensor (if needed for any further tensor-based operations).
    logits_tensor = torch.tensor(logits)

    print('Building predictions...')
    # For inference, we simply compute the predicted label as the argmax over logits.
    pred_labels = np.argmax(logits, axis=1)
    
    # Calculate probabilities via softmax.
    probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    
    # Create a DataFrame for segment-level predictions.
    # Here, we include the predicted label, logits, and computed probabilities.
    combined_results = pd.DataFrame({
        'y_pred': pred_labels,
        'logit_y0': logits[:, 0],
        'logit_y1': logits[:, 1],
        'p_class_0': probabilities[:, 0],
        'p_class_1': probabilities[:, 1]
    })    

    # Get the segment dataset with sequence_id (no ground truth column).
    # Adjust the select_columns call if your dataset API is different.
    segment_dataset_df = segment_dataset.select_columns(['sequence_id']).to_pandas()
    #segment_dataset_df['y'] = np.random.randint(0, 2, size=L)

    # Merge the predictions with the segment identifiers.
    # Assumes that the indices of segment_dataset_df and combined_results correspond to each other.
    merged_results = pd.concat([segment_dataset_df.reset_index(drop=True),
                                combined_results.reset_index(drop=True)], axis=1)

    print('Applying weighted voting...')
    # Aggregate segment-level predictions into sequence-level predictions using weighted voting.
    # Make sure that weighted_voting only depends on the columns available in merged_results.
    sequence_predictions = weighted_voting(merged_results)

    # Rename the aggregated probability columns if necessary.
    # (Assumes weighted_voting returns columns named 'score_class_0' and 'score_class_1'.)
    sequence_predictions.rename({'score_class_0': 'p_class_0', 'score_class_1': 'p_class_1'}, axis=1, inplace=True)

    # Map the aggregated numeric predictions to string labels.
    sequence_predictions['predicted_label'] = np.where(sequence_predictions['y_pred'] == 1, 'class_1', 'class_0')

    # Build the final table with just the desired columns.
    final_table = sequence_predictions[final_cols]

    return final_table