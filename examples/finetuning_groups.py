#!/usr/bin/env python3
"""
ProkBERT Finetuning with Length-Based Groups
=============================================

Follows the Colab notebook workflow (AutoTokenizer + AutoModelForSequenceClassification)
but reads a local FASTA file + labels.csv.

Input:
  - A FASTA file containing genome sequences
  - A labels.csv with columns: genome_id, label [, length, source]

Preprocessing:
  1. Load genomes from FASTA
  2. Cut each genome into contigs using a sliding window
  3. Assign each contig to a length group:
       Group A: 100-400 bp
       Group B: 400-800 bp
       Group C: 800-1200 bp
       Group D: 1200-1800 bp
  4. Tokenize with AutoTokenizer (same as Colab notebook)
  5. Train/eval split per genome (no data leakage)
  6. Finetune with HuggingFace Trainer

Usage:
    python finetuning_groups.py \
        --fasta /path/to/genomes.fasta \
        --labels_csv /path/to/labels.csv \
        --output_dir ./finetune_results \
        --model_name neuralbioinfo/prokbert-mini \
        --num_epochs 5 \
        --batch_size 128 \
        --bf16
"""

import argparse
import logging
import os
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Length group definitions
# ---------------------------------------------------------------------------

GROUPS = {
    "A": (100, 400),
    "B": (400, 800),
    "C": (800, 1200),
    "D": (1200, 1800),
}


# ---------------------------------------------------------------------------
# Data loading & sliding window segmentation
# ---------------------------------------------------------------------------

def load_labels_csv(path: str) -> Dict[str, int]:
    """Load labels CSV -> dict mapping genome_id -> label (int)."""
    df = pd.read_csv(path)
    if "genome_id" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"Labels CSV must have 'genome_id' and 'label' columns. Found: {list(df.columns)}"
        )
    label_map = dict(zip(df["genome_id"].astype(str), df["label"].astype(int)))
    logger.info("Loaded %d labels from %s", len(label_map), path)
    return label_map


def sliding_window_segments(
    sequence: str,
    window_size: int,
    step_size: int,
    min_length: int = 0,
) -> List[str]:
    """Cut a sequence into overlapping segments using a sliding window.

    Args:
        sequence: Nucleotide sequence string.
        window_size: Window length in bp.
        step_size: Step (stride) in bp.
        min_length: Discard segments shorter than this.

    Returns:
        List of segment strings.
    """
    segments = []
    seq_len = len(sequence)
    if seq_len < min_length:
        return segments

    for start in range(0, seq_len - min_length + 1, step_size):
        end = min(start + window_size, seq_len)
        seg = sequence[start:end]
        if len(seg) >= min_length:
            segments.append(seg)
        # Stop if we've reached the end
        if end == seq_len:
            break

    return segments


def build_segment_dataframe(
    fasta_path: str,
    label_map: Dict[str, int],
    groups: Dict[str, Tuple[int, int]],
    step_fraction: float = 0.5,
) -> pd.DataFrame:
    """Load FASTA, cut genomes into contigs via sliding window, assign to groups.

    For each length group (min_bp, max_bp):
      - window_size = max_bp
      - step_size = int(max_bp * step_fraction)
      - Only keep segments with length in [min_bp, max_bp]

    Args:
        fasta_path: Path to input FASTA file.
        label_map: genome_id -> label mapping.
        groups: Dict of group_name -> (min_bp, max_bp).
        step_fraction: Fraction of window_size used as step.

    Returns:
        DataFrame with columns: segment, genome_id, y, group
    """
    records = list(SeqIO.parse(fasta_path, "fasta"))
    logger.info("Loaded %d records from %s", len(records), fasta_path)

    all_rows = []
    missing = []

    for record in records:
        genome_id = record.id
        if genome_id not in label_map:
            missing.append(genome_id)
            continue
        label = label_map[genome_id]
        seq = str(record.seq).upper()

        for group_name, (min_bp, max_bp) in groups.items():
            window_size = max_bp
            step_size = max(1, int(max_bp * step_fraction))

            segments = sliding_window_segments(
                seq,
                window_size=window_size,
                step_size=step_size,
                min_length=min_bp,
            )
            for seg in segments:
                all_rows.append({
                    "segment": seg,
                    "genome_id": genome_id,
                    "y": label,
                    "group": group_name,
                })

    if missing:
        logger.warning(
            "%d genomes not found in labels CSV (first 5: %s)",
            len(missing), missing[:5],
        )

    df = pd.DataFrame(all_rows)
    logger.info("Total segments: %d", len(df))
    for g in sorted(groups.keys()):
        n = len(df[df["group"] == g])
        logger.info("  Group %s: %d segments", g, n)

    return df


# ---------------------------------------------------------------------------
# Train/val split by genome_id (no data leakage)
# ---------------------------------------------------------------------------

def split_by_genome(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train/val ensuring all segments from the same genome
    end up in the same split (no data leakage)."""
    genome_ids = df["genome_id"].unique()
    labels_per_genome = df.groupby("genome_id")["y"].first()

    train_ids, val_ids = train_test_split(
        genome_ids,
        test_size=test_size,
        random_state=seed,
        stratify=labels_per_genome.loc[genome_ids],
    )
    train_df = df[df["genome_id"].isin(set(train_ids))].reset_index(drop=True)
    val_df = df[df["genome_id"].isin(set(val_ids))].reset_index(drop=True)
    logger.info(
        "Split: %d train segments (%d genomes), %d val segments (%d genomes)",
        len(train_df), len(train_ids), len(val_df), len(val_ids),
    )
    return train_df, val_df


# ---------------------------------------------------------------------------
# Tokenization (same approach as Colab notebook)
# ---------------------------------------------------------------------------

def tokenize_function(examples, tokenizer):
    """Tokenize segments following the Colab notebook approach."""
    encoded = tokenizer.batch_encode_plus(
        examples["segment"],
        padding=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # Mask special tokens 2 ([CLS]) and 3 ([SEP]) as done in notebook
    mask_tokens = (input_ids == 2) | (input_ids == 3)
    attention_mask[mask_tokens] = 0

    y = torch.tensor(examples["y"], dtype=torch.int64)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": y,
    }


# ---------------------------------------------------------------------------
# Metrics (same as Colab notebook)
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)

    mcc = matthews_corrcoef(labels, predictions)
    acc = accuracy_score(labels, predictions)
    recall = recall_score(labels, predictions, average="weighted")

    try:
        if logits.shape[1] == 2:
            roc_auc = roc_auc_score(labels, logits[:, 1])
        else:
            roc_auc = roc_auc_score(labels, logits, multi_class="ovr")
    except ValueError:
        roc_auc = float("nan")

    return {
        "mcc": mcc,
        "accuracy": acc,
        "recall": recall,
        "roc_auc": roc_auc,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_group(
    group_name: str,
    train_ds: Dataset,
    val_ds: Dataset,
    model_name: str,
    output_dir: str,
    args: argparse.Namespace,
) -> Dict:
    """Train one group and return metrics."""
    logger.info("=" * 60)
    logger.info("GROUP %s — train: %d, val: %d", group_name, len(train_ds), len(val_ds))
    logger.info("=" * 60)

    group_output = os.path.join(output_dir, f"group_{group_name}")
    os.makedirs(group_output, exist_ok=True)

    # Load model fresh for each group
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, trust_remote_code=True
    )

    training_args = TrainingArguments(
        output_dir=group_output,
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="mcc",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        seed=args.seed,
        fp16=args.fp16,
        bf16=args.bf16,
        dataloader_num_workers=args.dataloader_num_workers,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Evaluate
    metrics = trainer.evaluate()
    logger.info("Group %s metrics: %s", group_name, metrics)

    # Save best model
    best_path = os.path.join(group_output, "best_model")
    trainer.save_model(best_path)
    logger.info("Saved best model to %s", best_path)

    return {"group": group_name, **metrics}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="ProkBERT Finetuning with Length-Based Groups"
    )
    parser.add_argument("--fasta", type=str, required=True, help="Path to input FASTA file")
    parser.add_argument("--labels_csv", type=str, required=True, help="Path to labels CSV")
    parser.add_argument(
        "--model_name", type=str, default="neuralbioinfo/prokbert-mini",
        help="Pretrained model name or path",
    )
    parser.add_argument("--output_dir", type=str, default="./finetune_results")
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=4e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.2, help="Fraction for validation")
    parser.add_argument("--step_fraction", type=float, default=0.5,
                        help="Sliding window step as fraction of window size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 (A100/H100)")
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument(
        "--groups", type=str, nargs="+", default=None,
        help="Specific groups to train (e.g. A B). Default: all (A B C D)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Save config
    with open(os.path.join(args.output_dir, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Select groups
    selected_groups = {k: v for k, v in GROUPS.items()}
    if args.groups:
        selected_groups = {k: GROUPS[k] for k in args.groups if k in GROUPS}
    logger.info("Groups to train: %s", list(selected_groups.keys()))

    # Load data
    label_map = load_labels_csv(args.labels_csv)

    logger.info("Building segments with sliding window (step_fraction=%.2f) ...", args.step_fraction)
    seg_df = build_segment_dataframe(
        args.fasta, label_map, selected_groups, step_fraction=args.step_fraction,
    )

    if len(seg_df) == 0:
        logger.error("No segments created. Check your FASTA and labels.")
        return

    # Split by genome
    train_df, val_df = split_by_genome(seg_df, test_size=args.test_size, seed=args.seed)

    # Load tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    num_cores = os.cpu_count() or 1

    # Train each group
    all_metrics = []
    for group_name in sorted(selected_groups.keys()):
        g_train = train_df[train_df["group"] == group_name].reset_index(drop=True)
        g_val = val_df[val_df["group"] == group_name].reset_index(drop=True)

        if len(g_train) == 0:
            logger.warning("Group %s has no training data, skipping.", group_name)
            continue
        if len(g_val) == 0:
            logger.warning("Group %s has no validation data, skipping.", group_name)
            continue

        logger.info("Tokenizing group %s (train=%d, val=%d) ...", group_name, len(g_train), len(g_val))

        # Convert to HuggingFace Dataset and tokenize
        train_hf = Dataset.from_pandas(g_train[["segment", "y"]])
        val_hf = Dataset.from_pandas(g_val[["segment", "y"]])

        train_tok = train_hf.map(
            lambda ex: tokenize_function(ex, tokenizer),
            batched=True, num_proc=num_cores, remove_columns=["segment"],
        )
        val_tok = val_hf.map(
            lambda ex: tokenize_function(ex, tokenizer),
            batched=True, num_proc=num_cores, remove_columns=["segment"],
        )

        metrics = train_group(
            group_name=group_name,
            train_ds=train_tok,
            val_ds=val_tok,
            model_name=args.model_name,
            output_dir=args.output_dir,
            args=args,
        )
        all_metrics.append(metrics)

    # Summary
    if all_metrics:
        summary_df = pd.DataFrame(all_metrics)
        summary_path = os.path.join(args.output_dir, "group_metrics.csv")
        summary_df.to_csv(summary_path, index=False)
        logger.info("Summary saved to %s", summary_path)
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
