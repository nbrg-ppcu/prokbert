"""
Binary sequence-classification training script for ProkBERT using a Hugging Face dataset.

Default dataset:
    neuralbioinfo/bacterial_promoters

Default column mapping:
    sequence_column = segment
    label_column = y

Default splits:
    train = train
    eval = test_sigma70
    test = test_multispecies

This script:
- loads a dataset directly from the Hugging Face Hub
- tokenizes DNA sequences with the ProkBERT tokenizer
- loads your local/custom ProkBERT sequence-classification head
- fine-tunes for binary classification
- reports accuracy, precision, recall, specificity, F1, balanced accuracy, and MCC

python test_binary_classification.py \
  --model_name_or_path neuralbioinfo/prokbert-mini \
  --output_dir ./runs/prokbert_bacterial_promoters \
  --num_train_epochs 1.2 \
  --per_device_train_batch_size 256 \
  --per_device_eval_batch_size 256 \
  --learning_rate 5e-4 \
  --max_length 85

# The expected result is sg like that
{'loss': 0.3634, 'grad_norm': 2.641960620880127, 'learning_rate': 3.82043935052531e-06, 'epoch': 1.19}                                                                                                    
{'eval_loss': 0.3191009759902954, 'eval_accuracy': 0.875, 'eval_precision': 0.8448087431693989, 'eval_recall': 0.8946759259259259, 'eval_specificity': 0.858, 'eval_f1': 0.8690275435637999, 'eval_balanced_accuracy': 0.876337962962963, 'eval_mcc': 0.7507947783112469, 'eval_runtime': 0.6337, 'eval_samples_per_second': 2941.608, 'eval_steps_per_second': 12.625, 'epoch': 1.2}                               
{'train_runtime': 214.3947, 'train_samples_per_second': 1248.68, 'train_steps_per_second': 4.884, 'train_loss': 0.4550933997291776, 'epoch': 1.2}                                                         
"""
from __future__ import annotations
import argparse
import inspect
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments, set_seed

try:
    from prokbert.models import ProkBertConfig, ProkBertForSequenceClassification
except ImportError:
    from models import ProkBertConfig, ProkBertForSequenceClassification


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Binary sequence-classification training script for ProkBERT using a HF dataset."
    )

    # dataset
    parser.add_argument("--dataset_name", type=str, default="neuralbioinfo/bacterial_promoters")
    parser.add_argument("--dataset_config", type=str, default="default")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--eval_split", type=str, default="test_sigma70")
    parser.add_argument("--test_split", type=str, default="test_multispecies")
    parser.add_argument("--sequence_column", type=str, default="segment")
    parser.add_argument("--label_column", type=str, default="y")

    # optional downsampling for fast tests
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)

    # model / io
    parser.add_argument("--model_name_or_path", type=str, default="neuralbioinfo/prokbert-mini")
    parser.add_argument("--output_dir", type=str, default="./prokbert_binary_hf_dataset")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--trust_remote_code", action="store_true", default=True)

    # training
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--classifier_dropout", type=float, default=0.1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")

    args = parser.parse_args()

    if args.fp16 and args.bf16:
        parser.error("Choose at most one of --fp16 or --bf16.")

    return args


def normalize_sequence(seq: Any) -> str:
    seq = str(seq).upper().replace("U", "T")
    seq = "".join(seq.split())
    return seq


def select_first_n(dataset: Dataset, n: int | None) -> Dataset:
    if n is None:
        return dataset
    n = min(int(n), len(dataset))
    return dataset.select(range(n))


def load_hf_splits(args: argparse.Namespace) -> DatasetDict:
    if args.dataset_config:
        raw = load_dataset(args.dataset_name, args.dataset_config)
    else:
        raw = load_dataset(args.dataset_name)

    required_splits = [args.train_split, args.eval_split]
    for split_name in required_splits:
        if split_name not in raw:
            available = ", ".join(raw.keys())
            raise ValueError(f"Required split '{split_name}' not found. Available splits: {available}")

    dataset_dict = DatasetDict(
        {
            "train": select_first_n(raw[args.train_split], args.max_train_samples),
            "validation": select_first_n(raw[args.eval_split], args.max_eval_samples),
        }
    )

    if args.test_split and args.test_split in raw:
        dataset_dict["test"] = select_first_n(raw[args.test_split], args.max_test_samples)
    elif args.test_split:
        available = ", ".join(raw.keys())
        raise ValueError(f"Requested test split '{args.test_split}' not found. Available splits: {available}")

    return dataset_dict


def build_label_mapping(train_dataset: Dataset, label_column: str) -> tuple[dict[Any, int], dict[int, str]]:
    unique_labels = sorted(set(train_dataset[label_column]))
    if len(unique_labels) != 2:
        raise ValueError(
            f"This script is for binary classification. Found labels: {unique_labels}"
        )
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: str(label) for label, idx in label2id.items()}
    return label2id, id2label


def prepare_datasets(
    raw_datasets: DatasetDict,
    tokenizer,
    args: argparse.Namespace,
    label2id: dict[Any, int],
    max_length: int,
) -> DatasetDict:
    seq_col = args.sequence_column
    label_col = args.label_column

    def preprocess(batch: dict[str, list[Any]]) -> dict[str, Any]:
        sequences = [normalize_sequence(x) for x in batch[seq_col]]
        tokenized = tokenizer(
            sequences,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        tokenized["labels"] = [label2id[x] for x in batch[label_col]]
        return tokenized

    tokenized = raw_datasets.map(
        preprocess,
        batched=True,
        desc="Tokenizing sequences",
    )

    keep_columns = {"input_ids", "attention_mask", "token_type_ids", "labels"}
    for split_name in tokenized.keys():
        remove_cols = [c for c in tokenized[split_name].column_names if c not in keep_columns]
        tokenized[split_name] = tokenized[split_name].remove_columns(remove_cols)

    return tokenized


def binary_classification_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]

    preds = np.argmax(logits, axis=-1)
    labels = np.asarray(labels)
    preds = np.asarray(preds)

    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    total = max(1, len(labels))
    accuracy = (tp + tn) / total
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    f1 = 0.0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)
    balanced_accuracy = 0.5 * (recall + specificity)

    denom = math.sqrt(max(1, (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom > 0 else 0.0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "balanced_accuracy": float(balanced_accuracy),
        "mcc": float(mcc),
    }


def build_training_arguments(args: argparse.Namespace) -> TrainingArguments:
    signature = inspect.signature(TrainingArguments.__init__)
    supported = set(signature.parameters)

    use_cuda = torch.cuda.is_available()
    fp16 = bool(args.fp16 and use_cuda)
    bf16 = bool(args.bf16 and use_cuda)

    kwargs: dict[str, Any] = {
        "output_dir": args.output_dir,
        "overwrite_output_dir": True,
        "do_train": True,
        "do_eval": True,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "logging_steps": args.logging_steps,
        "logging_strategy": "steps",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "mcc",
        "greater_is_better": True,
        "save_total_limit": args.save_total_limit,
        "dataloader_num_workers": args.dataloader_num_workers,
        "report_to": "none",
        "seed": args.seed,
        "remove_unused_columns": True,
        #"fp16": fp16,
        "bf16": True,
        "gradient_checkpointing": args.gradient_checkpointing,
    }

    if "eval_strategy" in supported:
        kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in supported:
        kwargs["evaluation_strategy"] = "epoch"

    kwargs = {k: v for k, v in kwargs.items() if k in supported}
    return TrainingArguments(**kwargs)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    raw_datasets = load_hf_splits(args)
    label2id, id2label = build_label_mapping(raw_datasets["train"], args.label_column)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )

    config = ProkBertConfig.from_pretrained(args.model_name_or_path)
    config.num_labels = 2
    config.problem_type = "single_label_classification"
    config.classifier_dropout = args.classifier_dropout
    config.label2id = {str(k): int(v) for k, v in label2id.items()}
    config.id2label = {int(k): v for k, v in id2label.items()}

    max_length = min(int(args.max_length), int(getattr(config, "max_position_embeddings", args.max_length)))

    model, loading_info = ProkBertForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        config=config,
        output_loading_info=True,
    )

    tokenized_datasets = prepare_datasets(
        raw_datasets=raw_datasets,
        tokenizer=tokenizer,
        args=args,
        label2id=label2id,
        max_length=max_length,
    )
    print(raw_datasets)
    print(tokenized_datasets)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None,
    )

    training_args = build_training_arguments(args)
    training_args = TrainingArguments(
        output_dir='fsdfds',
        report_to="none",
        logging_steps=20,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        num_train_epochs=2.0,
        bf16=True,
        #torch_compile=True,
        #torch_compile_mode ="max-autotune",
        #ddp_backend='nccl'
    )
    
    print(training_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=binary_classification_metrics,
    )

    print("=== Dataset sizes ===")
    for split_name, split_ds in tokenized_datasets.items():
        print(f"{split_name}: {len(split_ds)}")

    print("=== Label mapping ===")
    print(label2id)

    print("=== Loading info ===")
    print(json.dumps(loading_info, indent=2))

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.save_state()

    train_metrics = train_result.metrics
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)

    eval_metrics = trainer.evaluate(metric_key_prefix="eval")
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    if "test" in tokenized_datasets:
        test_metrics = trainer.evaluate(tokenized_datasets["test"], metric_key_prefix="test")
        trainer.log_metrics("test", test_metrics)
        trainer.save_metrics("test", test_metrics)

    with open(Path(args.output_dir) / "label_mapping.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "label2id": {str(k): int(v) for k, v in label2id.items()},
                "id2label": {str(k): v for k, v in id2label.items()},
            },
            handle,
            indent=2,
        )

    print(f"Saved model and metrics to: {args.output_dir}")


if __name__ == "__main__":
    main()
