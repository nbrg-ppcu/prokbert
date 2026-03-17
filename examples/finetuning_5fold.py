#!/usr/bin/env python3
"""
ProkBERT 5-Fold Cross-Validation Finetuning Script.

Expects pre-split data organized as:
    data_dir/
        labels.csv                 <-- genome_id,label,length,source
        fold_0/
            group_A_train.fasta, group_A_val.fasta
            group_B_train.fasta, group_B_val.fasta
            group_C_train.fasta, group_C_val.fasta
            group_D_train.fasta, group_D_val.fasta
        fold_1/
            ...
        fold_4/
            ...

FASTA header format (label and genome_id embedded in header):
    >NC_003315__contig_29 label=0 genome_id=NC_003315

Each genome may be split into multiple contigs for training.
The label and genome_id are parsed directly from the FASTA description.

Optionally, a labels CSV can be provided as fallback:
    genome_id,label,length,source
    NC_011421,1,132562,Dataset-1_virulent.fasta

Usage:
    python finetuning_5fold.py \
        --data_dir /path/to/folds \
        --model_name neuralbioinfo/prokbert-mini \
        --output_dir ./5fold_results \
        --num_epochs 10 \
        --batch_size 64 \
        --learning_rate 2e-5

    # With optional CSV fallback:
    python finetuning_5fold.py \
        --data_dir /path/to/folds \
        --labels_csv /path/to/labels.csv \
        ...
"""

import argparse
import json
import os
import re
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
)
from transformers import Trainer, TrainingArguments

from prokbert.sequtils import load_contigs, segment_sequence_contiguous
from prokbert.training_utils import (
    get_default_pretrained_model_parameters,
    get_torch_data_from_segmentdb_classification,
    compute_metrics_eval_prediction,
    check_nvidia_gpu,
)
from prokbert.models import BertForBinaryClassificationWithPooling
from prokbert.prok_datasets import ProkBERTTrainingDatasetPT
from prokbert import helper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

GROUPS = ["A", "B", "C", "D"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_labels_csv(labels_csv_path: str) -> Dict[str, int]:
    """Load labels CSV and return a dict mapping genome_id -> label (int).

    CSV format: genome_id,label,length,source
    Example:    NC_011421,1,132562,Dataset-1_virulent.fasta
    """
    df = pd.read_csv(labels_csv_path)
    if "genome_id" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"Labels CSV must have 'genome_id' and 'label' columns. "
            f"Found: {list(df.columns)}"
        )
    label_map = dict(zip(df["genome_id"].astype(str), df["label"].astype(int)))
    logger.info("Loaded %d labels from %s", len(label_map), labels_csv_path)
    return label_map


def parse_header_fields(description: str) -> Tuple[Optional[int], Optional[str]]:
    """Parse label and genome_id from a FASTA description line.

    Expected format:
        NC_003315__contig_29 label=0 genome_id=NC_003315

    Returns (label, genome_id) or (None, None) if not found.
    """
    label = None
    genome_id = None

    m_label = re.search(r"label=(\d+)", description)
    if m_label:
        label = int(m_label.group(1))

    m_gid = re.search(r"genome_id=(\S+)", description)
    if m_gid:
        genome_id = m_gid.group(1)

    return label, genome_id


def resolve_label(
    record_id: str,
    description: str,
    label_map: Optional[Dict[str, int]],
) -> Tuple[int, str]:
    """Resolve label for a FASTA record.

    Priority:
      1. Parse ``label=X`` directly from the FASTA description header.
      2. Fall back to CSV label_map using ``genome_id=X`` from header,
         then the record_id itself.

    Returns (label_int, genome_id).
    """
    header_label, header_genome_id = parse_header_fields(description)

    # Use genome_id from header if available, otherwise use record_id
    genome_id = header_genome_id or record_id

    # Priority 1: label embedded in header
    if header_label is not None:
        return header_label, genome_id

    # Priority 2: CSV lookup
    if label_map is not None:
        for key in (genome_id, record_id):
            if key in label_map:
                return label_map[key], genome_id

    raise KeyError(
        f"Cannot resolve label for '{record_id}' "
        f"(genome_id='{genome_id}'): not in header and not in labels CSV"
    )


def load_fasta_as_dataframe(
    fasta_path: str,
    label_map: Optional[Dict[str, int]] = None,
    max_segment_length: int = 2048,
    min_segment_length: int = 50,
) -> pd.DataFrame:
    """Load a FASTA file and return a DataFrame suitable for
    ``get_torch_data_from_segmentdb_classification``.

    Each FASTA record is a contig (e.g. ``NC_003315__contig_29``).
    Labels are parsed from the header (``label=0``) with optional CSV fallback.
    The ``genome_id`` from the header is used as ``sequence_id`` so that
    all contigs from the same genome share the same sequence_id.

    Columns: segment_id, segment, sequence_id, y, label
    """
    records = list(SeqIO.parse(fasta_path, "fasta"))
    if not records:
        raise ValueError(f"No records found in {fasta_path}")

    seg_params = {"min_length": min_segment_length, "max_length": max_segment_length}
    all_segments = []
    segment_id = 0
    missing_labels = []

    for record in records:
        seq = str(record.seq).upper()
        try:
            label_int, genome_id = resolve_label(
                record.id, record.description, label_map
            )
        except KeyError:
            missing_labels.append(record.id)
            continue
        label_str = f"class_{label_int}"

        # Segment long contigs; short ones become a single segment
        segments = segment_sequence_contiguous(seq, seg_params, sequence_id=genome_id)
        for seg in segments:
            all_segments.append(
                {
                    "segment_id": segment_id,
                    "segment": seg["segment"],
                    "sequence_id": genome_id,
                    "y": label_int,
                    "label": label_str,
                }
            )
            segment_id += 1

    if missing_labels:
        logger.warning(
            "%d contigs in %s have no label (first 5: %s)",
            len(missing_labels),
            fasta_path,
            missing_labels[:5],
        )

    return pd.DataFrame(all_segments)


def load_group_data(
    fold_dir: str,
    group: str,
    label_map: Optional[Dict[str, int]] = None,
    max_segment_length: int = 2048,
    min_segment_length: int = 50,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and val DataFrames for a single group within a fold."""
    train_path = os.path.join(fold_dir, f"group_{group}_train.fasta")
    val_path = os.path.join(fold_dir, f"group_{group}_val.fasta")

    if not os.path.isfile(train_path):
        raise FileNotFoundError(f"Missing train file: {train_path}")
    if not os.path.isfile(val_path):
        raise FileNotFoundError(f"Missing val file: {val_path}")

    train_df = load_fasta_as_dataframe(train_path, label_map, max_segment_length, min_segment_length)
    val_df = load_fasta_as_dataframe(val_path, label_map, max_segment_length, min_segment_length)

    logger.info(
        "  Group %s — train: %d segments (%d genomes), val: %d segments (%d genomes)",
        group,
        len(train_df),
        train_df["sequence_id"].nunique() if len(train_df) > 0 else 0,
        len(val_df),
        val_df["sequence_id"].nunique() if len(val_df) > 0 else 0,
    )
    return train_df, val_df


def load_fold_data(
    fold_dir: str,
    groups: List[str],
    label_map: Optional[Dict[str, int]] = None,
    max_segment_length: int = 2048,
    min_segment_length: int = 50,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Concatenate train/val data from all groups in a fold.

    Returns combined (train_df, val_df) with globally unique segment_ids.
    """
    train_dfs, val_dfs = [], []
    for group in groups:
        tr, va = load_group_data(fold_dir, group, label_map, max_segment_length, min_segment_length)
        train_dfs.append(tr)
        val_dfs.append(va)

    train_df = pd.concat(train_dfs, ignore_index=True)
    val_df = pd.concat(val_dfs, ignore_index=True)

    # Reassign globally unique segment_ids
    train_df["segment_id"] = range(len(train_df))
    val_df["segment_id"] = range(len(val_df))

    return train_df, val_df


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_detailed_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict:
    """Compute a comprehensive set of binary classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["roc_auc"] = -1.0

    if len(np.unique(y_true)) == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
        metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return metrics


# ---------------------------------------------------------------------------
# Training one fold
# ---------------------------------------------------------------------------

def train_one_fold(
    fold_idx: int,
    fold_dir: str,
    model_name: str,
    output_dir: str,
    groups: List[str],
    label_map: Optional[Dict[str, int]] = None,
    num_epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    max_segment_length: int = 2048,
    min_segment_length: int = 50,
    seed: int = 42,
    fp16: bool = False,
    gradient_accumulation_steps: int = 1,
    eval_strategy: str = "epoch",
    save_strategy: str = "epoch",
    load_best_model_at_end: bool = True,
    metric_for_best_model: str = "mcc",
) -> Dict:
    """Train and evaluate one fold. Returns evaluation metrics dict."""

    logger.info("=" * 60)
    logger.info("FOLD %d — %s", fold_idx, fold_dir)
    logger.info("=" * 60)

    # ---- Load data ----
    train_df, val_df = load_fold_data(
        fold_dir, groups, label_map, max_segment_length, min_segment_length
    )
    logger.info(
        "Fold %d total — train: %d, val: %d", fold_idx, len(train_df), len(val_df)
    )

    # ---- Model & tokenizer ----
    pretrained_model, tokenizer = get_default_pretrained_model_parameters(
        model_name=model_name,
        model_class="MegatronBertModel",
        output_hidden_states=False,
        output_attentions=False,
        move_to_gpu=False,
    )
    model = BertForBinaryClassificationWithPooling(pretrained_model)

    # ---- Tokenize ----
    logger.info("Tokenizing train set …")
    X_train, y_train, _ = get_torch_data_from_segmentdb_classification(tokenizer, train_df)
    logger.info("Tokenizing val set …")
    X_val, y_val, _ = get_torch_data_from_segmentdb_classification(tokenizer, val_df)

    train_ds = ProkBERTTrainingDatasetPT(X_train, y_train, AddAttentionMask=True)
    val_ds = ProkBERTTrainingDatasetPT(X_val, y_val, AddAttentionMask=True)

    # ---- Output dir ----
    fold_output = os.path.join(output_dir, f"fold_{fold_idx}")
    os.makedirs(fold_output, exist_ok=True)

    # ---- Training ----
    training_args = TrainingArguments(
        output_dir=fold_output,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        seed=seed,
        fp16=fp16,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics_eval_prediction,
    )

    trainer.train()

    # ---- Evaluate ----
    predictions = trainer.predict(val_ds)
    logits = predictions.predictions
    y_pred = np.argmax(logits, axis=-1)
    y_true = predictions.label_ids

    from scipy.special import softmax
    probs = softmax(logits, axis=-1)
    y_prob = probs[:, 1]

    fold_metrics = compute_detailed_metrics(y_true, y_pred, y_prob)
    fold_metrics["fold"] = fold_idx

    logger.info("Fold %d results: %s", fold_idx, json.dumps(fold_metrics, indent=2))

    # ---- Save model & metrics ----
    best_model_path = os.path.join(fold_output, "best_model")
    model.save_pretrained(best_model_path)
    logger.info("Model saved to %s", best_model_path)

    metrics_path = os.path.join(fold_output, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(fold_metrics, f, indent=2)

    # Save predictions
    pred_df = pd.DataFrame(
        {"y_true": y_true, "y_pred": y_pred, "prob_class_1": y_prob}
    )
    pred_df.to_csv(os.path.join(fold_output, "predictions.csv"), index=False)

    return fold_metrics


# ---------------------------------------------------------------------------
# Aggregate results
# ---------------------------------------------------------------------------

def aggregate_results(all_metrics: List[Dict], output_dir: str) -> pd.DataFrame:
    """Summarize metrics across all folds."""
    df = pd.DataFrame(all_metrics)
    summary = df.describe().T
    summary["metric"] = summary.index

    # Save per-fold and summary
    df.to_csv(os.path.join(output_dir, "per_fold_metrics.csv"), index=False)
    summary.to_csv(os.path.join(output_dir, "summary_metrics.csv"))

    logger.info("\n" + "=" * 60)
    logger.info("CROSS-VALIDATION SUMMARY")
    logger.info("=" * 60)

    metric_cols = [c for c in df.columns if c != "fold"]
    for col in metric_cols:
        vals = df[col].dropna()
        if len(vals) > 0:
            logger.info(
                "  %-20s  mean=%.4f  std=%.4f  min=%.4f  max=%.4f",
                col,
                vals.mean(),
                vals.std(),
                vals.min(),
                vals.max(),
            )

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="ProkBERT 5-Fold Cross-Validation Finetuning"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory containing fold_0/ … fold_4/ subdirectories",
    )
    parser.add_argument(
        "--labels_csv",
        type=str,
        default=None,
        help="Optional path to labels CSV (columns: genome_id,label,length,source). "
             "Only needed if FASTA headers do not contain label=X fields.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="neuralbioinfo/prokbert-mini",
        help="Pretrained model name or path (default: neuralbioinfo/prokbert-mini)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./5fold_results",
        help="Output directory for models and metrics",
    )
    parser.add_argument("--num_folds", type=int, default=5, help="Number of folds")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_segment_length", type=int, default=2048)
    parser.add_argument("--min_segment_length", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument(
        "--groups",
        type=str,
        nargs="+",
        default=GROUPS,
        help="Group names (default: A B C D)",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="*",
        default=None,
        help="Specific fold indices to run (default: all)",
    )
    parser.add_argument(
        "--metric_for_best_model",
        type=str,
        default="mcc",
        help="Metric for selecting best checkpoint (default: mcc)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    helper.set_seed(args.seed)
    check_nvidia_gpu()

    os.makedirs(args.output_dir, exist_ok=True)

    # Save run config
    with open(os.path.join(args.output_dir, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Load labels CSV if provided (optional fallback for headers without label=X)
    label_map = load_labels_csv(args.labels_csv) if args.labels_csv else None

    fold_indices = args.folds if args.folds is not None else list(range(args.num_folds))

    all_metrics = []
    for fold_idx in fold_indices:
        fold_dir = os.path.join(args.data_dir, f"fold_{fold_idx}")
        if not os.path.isdir(fold_dir):
            logger.warning("Fold directory not found: %s — skipping", fold_dir)
            continue

        fold_metrics = train_one_fold(
            fold_idx=fold_idx,
            fold_dir=fold_dir,
            model_name=args.model_name,
            output_dir=args.output_dir,
            groups=args.groups,
            label_map=label_map,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            max_segment_length=args.max_segment_length,
            min_segment_length=args.min_segment_length,
            seed=args.seed,
            fp16=args.fp16,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            metric_for_best_model=args.metric_for_best_model,
        )
        all_metrics.append(fold_metrics)

    if all_metrics:
        aggregate_results(all_metrics, args.output_dir)
    else:
        logger.warning("No folds were processed!")


if __name__ == "__main__":
    main()
