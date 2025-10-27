import torch
import os
from ncbi_dataset import NCBI_dataset
from prokbert.tokenizer import LCATokenizer
from datetime import datetime
from prokbert.models import *

from transformers import TrainingArguments, Trainer, EarlyStoppingCallback, TrainerCallback
from transformers import AdamW
from transformers import get_scheduler
from transformers import default_data_collator
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
)

def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #directory for NCBI dataset
    basedir = "/leonardo_scratch/fast/EUHPC_R04_194/training_datasets/NCBI/experiment"
    batch_size = 64
    dataset = NCBI_dataset(basedir, batch_size, Ls=1024)
    tokenizer = LCATokenizer(kmer=6, shift=2, vocab_file = "/leonardo_work/EUHPC_R04_194/prokbert/src/prokbert/data/prokbert_vocabs/prokbert-base-dna6/vocab.txt")

    num_of_classes = len(dataset.unique_assembly)

    #old ProkBert model 
    bert_model_path = "neuralbioinfo/prokbert-mini-long"
    model = ProkBertForCurricularClassification.from_pretrained(
            bert_model_path,
            bert_base_model = bert_model_path,
            torch_dtype=torch.bfloat16,
            curricular_num_labels = num_of_classes,
            curricular_face_m = 0.5,
            curricular_face_s = 64.0,
            classification_dropout_rate = 0.1,
            curriculum_hidden_size = 128,
        )

    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of params: {num_params}")

    print("Set up learning utilities")
    bert_params = []
    head_params = []

    for name, param in model.named_parameters():
        if "model" in name:
            bert_params.append(param)
        else:
            head_params.append(param)

    optimizer = AdamW([
        {'params': bert_params, 'lr': 0.0001},
        {'params': head_params, 'lr': 0.001}
    ])

    num_warmup = 100
    scheduler = get_scheduler(
        "cosine",
        optimizer = optimizer,
        num_warmup_steps = num_warmup,
        num_training_steps = 500
    )
    max_steps = 500
    def collate_fn(features): 
        return features[0]
    data_collator = collate_fn

    
    gradient_accumulation_steps = 1
    num_train_epochs = 1

    #directory where all training artifacts will be saved
    basemodel_output_dir = "/leonardo_work/EUHPC_R04_194/prokbinner/models"
    current_date = str(datetime.now().strftime("%Y-%m-%d"))
    finetuned_model_name = f'prokbert_NCBI__bs{batch_size*gradient_accumulation_steps}__e{num_train_epochs}_{current_date}'
    finetune_model_path = os.path.join(basemodel_output_dir, finetuned_model_name)
    loggingdir_path = os.path.join(basemodel_output_dir, finetuned_model_name, "tensorboard_logs")

    training_args = TrainingArguments(
        output_dir = finetune_model_path,
        overwrite_output_dir = True,
        logging_strategy = "steps",
        per_device_train_batch_size = batch_size,
        gradient_accumulation_steps = gradient_accumulation_steps,
        max_steps = max_steps, 
        weight_decay = 0.00001,
        logging_steps = 5,
        report_to="tensorboard",
        logging_dir=loggingdir_path,
        dataloader_num_workers = 8,
        dataloader_prefetch_factor = 4,
        torch_compile = False,
        bf16 = True,
        save_total_limit = 1,              # limit the total amount of checkpoints
        save_steps = 100,
        load_best_model_at_end = False,
        max_grad_norm = 1.0,  # <- this enables gradient clipping!
        ddp_find_unused_parameters = True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        optimizers=(optimizer, scheduler),
    )
    
    trainer.train()

if __name__ == "__main__":
    main()