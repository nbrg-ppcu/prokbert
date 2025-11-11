import torch
import os
import random
from torch.utils.data import DataLoader

from datasets import load_from_disk
from prokbert.models import *
from transformers import AdamW

from datasets import DatasetDict
from prokbert.tokenizer import LCATokenizer
from transformers import DataCollatorWithPadding
from transformers import get_scheduler
from transformers import TrainingArguments, Trainer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap
from transformers import TrainerCallback
from transformers import default_data_collator
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau


from transformers import AutoTokenizer, AutoModel

def get_model_embeddings_umap(grouped_means, labels, seq_ids, plot_path,
                                plot_name = "umap_embedding_visualization.png",
                                umap_n_neighbors=30,
                                umap_min_dist = 0.001,
                                umap_random_state=42,
                                save_plot=True,
                                plot_point_size=1):
    umap_n_neighbors = int(np.log(grouped_means.shape[0]))
    reducer = umap.UMAP(random_state=umap_random_state, n_neighbors = umap_n_neighbors, min_dist = umap_min_dist, metric='cosine', n_jobs=-1, spread=5, n_components=2)
    umap_embeddings = reducer.fit_transform(grouped_means)

    umap_df = pd.DataFrame(data={'umap_1': umap_embeddings[:, 0], 'umap_2': umap_embeddings[:, 1],
    "labels": labels, "sequence_id": seq_ids})


    plt.figure(figsize=(20, 7))
    g = sns.scatterplot(umap_df, x="umap_1", y="umap_2",hue="labels", palette="tab20", s=plot_point_size)
    plt.subplots_adjust(top=0.9)
    g.get_legend().set_visible(False)
    plt.suptitle(f'Visualization of embeddings')
    plt.savefig(os.path.join(plot_path, plot_name))
    return umap_df

def create_embeddings(model, dataloader, output_name,  plot_name):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    representations = []
    labels = []
    seq_ids = []
    batch_ind = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            columns_to_keep = ['input_ids', 'attention_mask']
            labels.append(batch["labels"])
            seq_ids.append(batch["sequence_id"])
            batch = {k: v.to(device) for k, v in batch.items() if k in columns_to_keep} 
            with torch.no_grad():
                mean_pooled = model(**batch)
                representations_batch = mean_pooled.detach().cpu().float().numpy()
                representations.append(representations_batch)
            batch_ind += 1
    model.train()    
    embeddings = np.concatenate(representations)
    labels = np.concatenate(labels)
    seq_ids = np.concatenate(seq_ids)

    df = pd.DataFrame(embeddings)
    df['group'] = seq_ids

    grouped_means = df.groupby('group').mean().values

    df['labels'] = labels
    labels_pooled = df[['labels', 'group']].groupby('group').first().values[:, 0]
    seq_ids_pooled = df.groupby('group')['group'].agg('first').values
    umap_df = get_model_embeddings_umap(embeddings, labels, seq_ids,
                                        plot_path = output_name,
                                        plot_point_size=50,
                                        plot_name = plot_name)
    return umap_df

class PlottingCallback(TrainerCallback):
    def __init__(self, output_dir, eval_dataset, plot_step):
        self.num_evaluations = 0
        self.plot_step = plot_step
        self.output_dir = output_dir
        self.eval_dataloader = DataLoader(eval_dataset, batch_size=128, shuffle=False, collate_fn=default_data_collator)
        
    def on_evaluate(self, args, state, control, model=None,  **kwargs):
        if self.num_evaluations % self.plot_step == 0:
            create_embeddings(model, self.eval_dataloader, self.output_dir,  f"./eval_{self.num_evaluations}_emb.png")
        self.num_evaluations += 1

def finetune(output_folder_path):
    num_device = 4
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    basedir = "/leonardo_scratch/fast/EUHPC_R04_194/training_datasets/gtdb/toy_dataset1024_bacterias/"
    batch_size = 64
    dataset = load_from_disk(basedir)
    dataset_training = dataset['train']
    dataset_test = dataset['validation']
    dataset_new_genome = dataset['test']

    num_classes = 10
    allowed_labels = dataset_test.unique('labels')[:num_classes]

    dataset_training = dataset_training.filter(
        lambda batch: [label in allowed_labels for label in batch["labels"]],
        batched=True
    )
    dataset_test = dataset_test.filter(
        lambda batch: [label in allowed_labels for label in batch["labels"]],
        batched=True
    )
    print("After filtering: ")
    print(f"Size of train {len(dataset_training)}")
    print(f"Size of test {len(dataset_test)}")
    print(f"Number of unique classes {len(allowed_labels )}")
    unique_cats = dataset_training.unique('labels')
    cat2id = {cat: i for i, cat in enumerate(unique_cats)}

    def encode_batch(batch):
        batch['full_labels'] = batch['labels']
        batch['labels'] = [cat2id[c] for c in batch['labels']]
        return batch

    dataset_training = dataset_training.map(encode_batch, batched=True)
    dataset_test = dataset_test.map(encode_batch, batched=True)

    num_of_classes = len(unique_cats)
    print(f"Size of train {len(dataset_training)}")
    print(f"Number of unique classes {num_of_classes}")

    num_classes = 10
    allowed_labels = dataset_new_genome.unique('labels')[:num_classes]

    dataset_new_genome = dataset_new_genome.filter(
        lambda batch: [label in allowed_labels for label in batch["labels"]],
        batched=True
    )
    unique_cats = dataset_new_genome.unique('labels')
    cat2id = {cat: i for i, cat in enumerate(unique_cats)}

    def encode_batch(batch):
        batch['full_labels'] = batch['labels']
        batch['labels'] = [cat2id[c] for c in batch['labels']]
        return batch

    dataset_new_genome = dataset_new_genome.map(encode_batch, batched=True)

    test_dataloader = DataLoader(dataset_new_genome, batch_size=128, shuffle=False, collate_fn=default_data_collator)

    bert_model_path = "neuralbioinfo/prokbert-mini-long"
    model = ProkBertForCurricularClassification.from_pretrained(
        bert_model_path,
        bert_base_model = bert_model_path,
        torch_dtype=torch.bfloat16,
        curricular_num_labels = num_classes,
        curricular_face_m = 0.75,
        curricular_face_s = 64.0,
        classification_dropout_rate = 0.1,
        curriculum_hidden_size = 128,
    )

    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of params of the model: {num_params}")

    print("Set up learning utilities")
    bert_params = []
    head_params = []

    for name, param in model.named_parameters():
        if "bert" in name:
            #param.requires_grad = False
            bert_params.append(param)
        else:
            head_params.append(param)

    optimizer = AdamW([
        {'params': head_params, 'lr': 0.004},
        {'params': bert_params, 'lr': 0.0001}
    ])
    print(f"Num trainable params: {sum(p.numel() for p in head_params)}")
    print(f"Num trainable params: {sum(p.numel() for p in bert_params)}")

    num_epoches = 20
    num_warmup = 0
    count = len(dataset_training)
    max_steps = int((count // (batch_size * 4)) * num_epoches)
    eval_step = (max_steps // 10) // 10
    tokenizer = LCATokenizer(kmer=6, shift=2, vocab_file = "/leonardo_work/EUHPC_R04_194/prokbert/src/prokbert/data/prokbert_vocabs/prokbert-base-dna6/vocab.txt")

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",    
        factor=0.1,  
        patience=5,
        verbose=True
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    output_model_path = output_folder_path #path to save trained models
    training_args = TrainingArguments(
        output_dir=output_model_path,
        eval_strategy="steps",
        overwrite_output_dir = False,
        logging_strategy = "steps",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        max_steps = max_steps, 
        weight_decay=0.0,
        logging_steps=10,
        report_to=None,
        eval_steps = 100,
        eval_accumulation_steps=1,
        dataloader_num_workers=1,
        dataloader_prefetch_factor=1,
        torch_compile=False,
        bf16=True,
        save_total_limit=1,
        save_steps = max_steps,
        load_best_model_at_end=True,
        max_grad_norm=1.0,
        ddp_find_unused_parameters=True,
        ddp_backend="nccl",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_training,
        eval_dataset=dataset_test,
        processing_class=tokenizer,
        data_collator=data_collator,
        optimizers=(optimizer, scheduler),
        callbacks=[PlottingCallback(output_model_path, dataset_test, plot_step=eval_step)],
    )
    trainer.evaluate()
    trainer.train()

    #create embeddings for new genomes
    create_embeddings(model, test_dataloader, "/leonardo_work/EUHPC_R04_194/test_prokbert/", f"new_genomes_emb_curr.png")
    
    #compare with Prokbert embeddings
    model_name_path = 'neuralbioinfo/prokbert-mini-long'
    model = AutoModel.from_pretrained(model_name_path, trust_remote_code=True)


if __name__ == "__main__":
    output_path = "./test_prokbert"
    os.makedirs(output_path, exist_ok=True)
    finetune(output_path)