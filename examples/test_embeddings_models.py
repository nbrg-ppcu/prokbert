
# This is a simple script for testing the embendinng generation with different models

from dataclasses import dataclass
import os
from typing import Dict, Tuple

import numpy as np
import torch
from datasets import ClassLabel, load_dataset
from sklearn.metrics import accuracy_score
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)
import pandas as pd
import os
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from datasets import load_dataset
import seaborn as sns
import matplotlib.pyplot as plt
import umap

from prokbert.models import ProkBertModel



@dataclass
class Config:
    seed: int = 42
    dataset_name: str = "neuralbioinfo/ESKAPE-genomic-features"
    dataset_split: str = "ESKAPE"
    model_name: str = "neuralbioinfo/prokbert-mini-long"
    output_dir: str = "./"
    max_length: int = 256
    eval_batch_size: int = 256


def main():
    print('Testing the EMBEDDING generation')

    cfg = Config()
    set_seed(cfg.seed)

    model_name_path = cfg.model_name

    tokenizer = AutoTokenizer.from_pretrained(model_name_path, trust_remote_code=True)
    # We are going to use base, encoder model
    model = ProkBertModel.from_pretrained(model_name_path)

    #print(model)

    print('Cica')

    dataset = load_dataset("neuralbioinfo/ESKAPE-genomic-features", split='ESKAPE')
    dataset.shuffle(seed=cfg.seed)    
    dataset_sample = dataset.select(range(1000))
    
    def truncate_segment(example):
        example["segment"] = example["segment"][:10]
        return example

    #dataset_sample = dataset_sample.map(truncate_segment)
    print(dataset_sample) 
    num_cores = os.cpu_count()



    def tokenize_function(examples):
        
        return tokenizer(
            examples["segment"],  # Replace 'sequence' with the actual column name if different
            padding=True,
            truncation=True,
            max_length=512,  # Set the maximum sequence length if needed
            return_tensors="pt"
        )

    # Apply tokenization
    tokenized_dataset = dataset_sample.map(tokenize_function, batched=True, num_proc=num_cores)

    print(tokenized_dataset[0:2])

    training_args = TrainingArguments(
        output_dir="./results",  # Output directory
        per_device_eval_batch_size=16,  # Batch size for evaluation
        remove_unused_columns=True,  # Ensure compatibility with input format
        report_to="none",  # No reporting needed
    )

    def summarize_weights(model):
        stats = {}
        for name, param in model.named_parameters():
            t = param.detach().cpu()
            stats[name] = {
                "shape": tuple(t.shape),
                "mean": float(t.mean()),
                "std": float(t.std()),
                "sum": float(t.sum()),
            }
        return stats


    stats = summarize_weights(model)
    for k, v in stats.items():
        print(k, v)    


    # Set up the Trainer for prediction and evaluation
    trainer = Trainer(
        model=model,  # Dummy model
        args=training_args,  # Evaluation arguments
    )

    predictions = trainer.predict(tokenized_dataset)
    print(predictions.predictions)
    print(predictions.predictions.shape)
    last_hidden_states = predictions.predictions[0]
    last_hidden_states = predictions.predictions

    print(last_hidden_states)
    print(last_hidden_states.shape)

    representations = last_hidden_states.mean(axis=1)

    #predictions.last_hidden_state
    umap_random_state = 42
    n_neighbors=20
    min_dist = 0.4
    reducer = umap.UMAP(random_state=umap_random_state, n_neighbors=n_neighbors, min_dist=min_dist)
    print('Running UMAP ....')
    umap_embeddings = reducer.fit_transform(representations)

    dataset_df = dataset_sample.to_pandas()
    dataset_df['umap_1'] = umap_embeddings[:, 0]
    dataset_df['umap_2'] = umap_embeddings[:, 1]

    g = sns.FacetGrid(
        dataset_df,
        col="strand",
        hue="class_label",
        palette="Set1",
        height=6,
    )
    g.map_dataframe(sns.scatterplot, x="umap_1", y="umap_2")
    g.add_legend()

    os.makedirs(cfg.output_dir, exist_ok=True)
    output_path = os.path.join(cfg.output_dir, "umap_embeddings.png")
    g.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to: {output_path}")

    plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()