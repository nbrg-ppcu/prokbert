#!/usr/bin/env python3
# Curriculum example for finetuning with the ESKAPE dataset.

from dataclasses import dataclass
import os
from typing import Dict, Tuple

import numpy as np
import torch
from datasets import ClassLabel, load_dataset
from sklearn.metrics import accuracy_score
from transformers import (
    AdamW,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from prokbert.curriculum_utils import plot_umap_embeddings
from prokbert.models import ProkBertForCurricularClassification

@dataclass
class Config:
    dataset_name: str = "neuralbioinfo/ESKAPE-genomic-features"
    dataset_split: str = "ESKAPE"
    model_name: str = "neuralbioinfo/prokbert-mini-long"
    output_dir: str = "./curriculum_finetuning_outputs"
    max_length: int = 256
    train_batch_size: int = 256
    eval_batch_size: int = 256
    num_train_epochs: int = 2
    gradient_accumulation_steps: int = 1
    seed: int = 42
    test_size: float = 0.2
    max_samples: int = 0
    curricular_face_m: float = 0.5
    curricular_face_s: float = 64.0
    curriculum_hidden_size: int = 128
    classification_dropout_rate: float = 0.1
    backbone_lr: float = 1.6e-5
    head_lr: float = 4.8e-4
    beta1: float = 0.5794
    beta2: float = 0.6576
    weight_decay: float = 0
    logging_steps: int = 25


def resolve_columns(dataset) -> Tuple[str, str]:
    sequence_candidates = ("segment", "sequence", "seq")
    label_candidates = ("contig_id", "class_label", "label", "labels", "y")

    sequence_col = next((c for c in sequence_candidates if c in dataset.column_names), None)
    label_col = next((c for c in label_candidates if c in dataset.column_names), None)

    if sequence_col is None:
        raise ValueError("No sequence column found. Expected one of: segment, sequence, seq.")
    if label_col is None:
        raise ValueError("No label column found. Expected one of: class_label, label, labels, y.")

    return sequence_col, label_col


def encode_labels(dataset, label_col: str) -> Tuple[Dict[int, str], Dict[str, int]]:
    if not isinstance(dataset.features[label_col], ClassLabel):
        dataset = dataset.class_encode_column(label_col)

    label_feature = dataset.features[label_col]
    id2label = {i: name for i, name in enumerate(label_feature.names)}
    label2id = {name: i for i, name in id2label.items()}
    return dataset, id2label, label2id


def tokenize_dataset(dataset, tokenizer, sequence_col: str, max_length: int):
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples[sequence_col],
            truncation=True,
            max_length=max_length,
        )
        tokenized["labels"] = examples["labels"]
        if "sequence_id" in examples:
            tokenized["sequence_id"] = examples["sequence_id"]
        return tokenized

    remove_columns = [c for c in dataset.column_names if c not in ("labels", "sequence_id")]
    num_proc = min(os.cpu_count() or 1, 8)
    return dataset.map(tokenize_function, batched=True, remove_columns=remove_columns, num_proc=num_proc)


def compute_metrics(eval_pred):
    logits = eval_pred.predictions if hasattr(eval_pred, "predictions") else eval_pred[0]
    labels = eval_pred.label_ids if hasattr(eval_pred, "label_ids") else eval_pred[1]
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}

def main():
    cfg = Config()
    set_seed(cfg.seed)

    dataset = load_dataset(cfg.dataset_name, split=cfg.dataset_split)
    dataset = dataset.shuffle(seed=cfg.seed)
    sequence_col, label_col = resolve_columns(dataset)
    dataset = dataset.remove_columns(
        [col for col in dataset.column_names if col not in (sequence_col, label_col)]
    )
    print(dataset)

    #1/0

    if cfg.max_samples:
        dataset = dataset.select(range(min(cfg.max_samples, len(dataset))))

    dataset = dataset.add_column("sequence_id", list(range(len(dataset))))

    dataset, id2label, label2id = encode_labels(dataset, label_col)

    if label_col != "labels":
        dataset = dataset.rename_column(label_col, "labels")

    print(dataset)
    split = dataset.train_test_split(test_size=cfg.test_size, seed=cfg.seed)
    temp = split["test"].train_test_split(test_size=0.5, seed=cfg.seed)
    train_ds = split["train"]
    eval_ds = temp["train"]
    test_ds = temp["test"]

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    train_ds = tokenize_dataset(train_ds, tokenizer, sequence_col, cfg.max_length)
    eval_ds = tokenize_dataset(eval_ds, tokenizer, sequence_col, cfg.max_length)
    test_ds = tokenize_dataset(test_ds, tokenizer, sequence_col, cfg.max_length)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model_dtype = torch.bfloat16 if use_bf16 else torch.float32

    testmodel = ProkBertForCurricularClassification.from_pretrained(cfg.output_dir + '_post')
    
    testmodel.eval()


    model = ProkBertForCurricularClassification.from_pretrained(
        cfg.model_name,
        curricular_num_labels=len(id2label),
        curricular_face_m=cfg.curricular_face_m,
        curricular_face_s=cfg.curricular_face_s,
        classification_dropout_rate=cfg.classification_dropout_rate,
        curriculum_hidden_size=cfg.curriculum_hidden_size,
        torch_dtype=model_dtype,
        id2label=id2label,
        label2id=label2id,
    )
    model.save_pretrained(cfg.output_dir + '_init')


    backbone_params = [p for n, p in model.named_parameters() if n.startswith("bert.")]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("bert.")]

    optimizer = AdamW(
        [
            {"params": backbone_params, "lr": cfg.backbone_lr},
            {"params": head_params, "lr": cfg.head_lr},
        ],
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    testmodel = testmodel.to(device)

    print(train_ds)
    plot_umap_embeddings(
        model,
        train_ds,
        data_collator,
        cfg.output_dir,
        "umap_before_training.png",
        cfg.eval_batch_size,
        cfg.seed,
    )
    print('Loading existing model .... ')
    plot_umap_embeddings(
        testmodel,
        train_ds,
        data_collator,
        cfg.output_dir,
        "umap_done_after_training.png",
        cfg.eval_batch_size,
        cfg.seed,
    )



    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        overwrite_output_dir=False,
        report_to="none",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=cfg.logging_steps,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        bf16=use_bf16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None),
    )

    trainer.train()
    plot_umap_embeddings(
        model,
        train_ds,
        data_collator,
        cfg.output_dir,
        "umap_after_training.png",
        cfg.eval_batch_size,
        cfg.seed,
    )
    metrics = trainer.evaluate(test_ds)
    print(metrics)
    model.save_pretrained(cfg.output_dir + '_post')

    print('Loading the existing model just for test')


    


if __name__ == "__main__":
    main()
