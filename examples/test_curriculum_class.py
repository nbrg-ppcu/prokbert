"""
torchrun --nproc_per_node=1 test_curriculum_class.py


"""

import torch
USE_TF32 = False   # safest while debugging; set True if you want TF32 on fp32 ops
torch.backends.cuda.matmul.allow_tf32 = USE_TF32
torch.backends.cudnn.allow_tf32 = USE_TF32
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from prokbert.sequtils import *
from prokbert.training_utils import *
from prokbert.models import ProkBertForCurricularClassification
from prokbert.tokenizer import LCATokenizer
from prokbert.curriculum_utils import compute_umap_for_dataset, evaluate_embeddings
from datasets import Dataset, load_dataset, ClassLabel
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from os.path import join
import os
from prokbert.trainers.curriculum_trainer import CustomTrainer
from torch.optim import AdamW


REPO_ID = "neuralbioinfo/eskapee"
MODEL_NAME = "neuralbioinfo/prokbert-mini-long"

OUTPUT_PATH = "./test_megatroncurr"


def reset_matmul_precision(tf32: bool = False) -> None:
    mode = "high" if tf32 else "highest"
    torch.set_float32_matmul_precision(mode)

    # Optional hard reset in case another import changed backend-specific state.
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
            torch.backends.cuda.matmul.fp32_precision = "tf32" if tf32 else "ieee"

    if hasattr(torch.backends, "mkldnn") and hasattr(torch.backends.mkldnn, "matmul"):
        if hasattr(torch.backends.mkldnn.matmul, "fp32_precision"):
            torch.backends.mkldnn.matmul.fp32_precision = "tf32" if tf32 else "ieee"




def main() -> None:

    print('Just briefly testing the curriculum class method')

    # 1) Load dataset
    ds = load_dataset(REPO_ID, split="train")

    ds = ds.add_column("assembly_label", ds["assembly"])
    ds = ds.class_encode_column("assembly_label")          # now `assembly_label` is an integer-backed ClassLabel
    ds = ds.rename_column("assembly_label", "label_id")    # trainer-friendly name

    # 4) Extract mappings (id2label / label2id)
    label_feature = ds.features["label_id"]
    id2label = dict(enumerate(label_feature.names))
    label2id = {name: i for i, name in id2label.items()}

    print(f"Number of assembly labels: {len(id2label)}")



    # Converting the dataset into a pandas dataframe for further dataprocessing
    seed = 42
    test_size = 0.10
    train_batch_size = 64
    eval_batch_size = 32
    num_train_epochs = 2.5
    num_eval_steps = 400
    use_bf16 = True
    seed = 42
    test_size = 0.10
    max_length = 1000
    max_length = 2000


    sequences = ds.to_pandas()
    lut_cols = ["sequence_id", "label_id"]


    print("[prepare_dataset] Running segmentation")
    segmentation_params = {
        "max_length": max_length,
        "min_length": int(max_length * 0.5),
        "type": "contiguous",
    }
    raw_segment_df = segment_sequences(
        sequences, segmentation_params, AsDataFrame=True
    )
    print(f"[prepare_dataset] Number of segments: {len(raw_segment_df)}")


    for extra in ["assembly", "taxon", "taxon_short", "taxon_name"]:
        if extra in sequences.columns and extra not in lut_cols:
            lut_cols.append(extra)
            
    label_lut = sequences[lut_cols].drop_duplicates(subset=["sequence_id"])
    if "sequence_id" not in raw_segment_df.columns:
        raise ValueError(
            f"`raw_segment_df` has no `sequence_id` column. Available columns: {raw_segment_df.columns.tolist()}"
        )

    raw_segment_df = raw_segment_df.merge(
        label_lut,
        on="sequence_id",
        how="left",
        validate="many_to_one",
    )
    raw_segment_df = raw_segment_df.sample(frac=1.0)
    raw_segment_df["label_id"] = raw_segment_df["label_id"].astype("int64")
    hf_dataset = Dataset.from_pandas(raw_segment_df, preserve_index=False)
    hf_dataset = hf_dataset.rename_column("label_id", "labels")

    split = hf_dataset.train_test_split(test_size=test_size, seed=seed, shuffle=True)
    train_ds = split["train"]
    test_ds = split["test"]

    use_bf16 = True
    model_dtype = torch.bfloat16 if use_bf16 else torch.float32

    num_labels = len(id2label)

    model, loading_info = ProkBertForCurricularClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        problem_type="single_label_classification",
        classifier_head_type="curricular",
        classifier_pooling="attention",      # change to "cls" if you want CLS pooling
        classifier_dropout=0.1,
        curricular_embedding_size=256,       # None or -1 => no projection layer
        curricular_margin=0.5,
        curricular_scale=64.0,
        output_loading_info=True,
    )
    print("Missing keys:", loading_info["missing_keys"])
    print("Unexpected keys:", loading_info["unexpected_keys"])

    tokenizer = LCATokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    num_cores = max(1, min(os.cpu_count() or 1, 16))  

    def _tokenize_fn(batch):
        tok = tokenizer(
            batch["segment"],
            padding=False,              
            truncation=True,
            max_length=max_length,
        )
        masks = tok["attention_mask"]
        # Keep labels (and any identifiers you want)
        tok["labels"] = batch["labels"]
        if "sequence_id" in batch:
            tok["sequence_id"] = batch["sequence_id"]
        if "segment_id" in batch:
            tok["segment_id"] = batch["segment_id"]
        return tok

    # Drop heavy text columns after tokenization, but keep identifiers + labels
    keep_cols = {"labels", "sequence_id", "segment_id"}
    remove_cols = [c for c in hf_dataset.column_names if c not in keep_cols and c != "segment"]

    print(f"[prepare_dataset] Tokenizing with {num_cores} CPU core(s)")
    tokenized_train_ds = train_ds.map(
        _tokenize_fn,
        batched=True,
        num_proc=num_cores,
        remove_columns=remove_cols + ["segment"],  # remove raw text segment post-tokenization
        keep_in_memory=True,
        desc="Tokenize segments",
    )

    print(f"[prepare_dataset] Tokenizing with {num_cores} CPU core(s)")
    tokenized_test_ds = test_ds.map(
        _tokenize_fn,
        batched=True,
        num_proc=num_cores,
        remove_columns=remove_cols + ["segment"],  # remove raw text segment post-tokenization
        keep_in_memory=True,
        desc="Tokenize segments",
    )

    umap_n = 2000
    umap_seed = 123

    umap_ds = tokenized_test_ds.shuffle(seed=umap_seed)#.select(range(min(umap_n, len(tokenized_train_ds))))
    emb, coords = compute_umap_for_dataset(
        model=model,
        dataset=umap_ds,
        data_collator=data_collator,
        batch_size=eval_batch_size,
        seed=42,
    )
    score_before = evaluate_embeddings(emb, umap_ds["labels"])
    print(f"Silhouette score before training: {score_before:.4f}")


    #coords are aligned with umap_ds order
    plot_df = pd.DataFrame(coords, columns=["umap_1", "umap_2"])
    meta_cols = ["sequence_id", "segment_id", "assembly", "taxon", "taxon_short", "taxon_name"]
    meta_df = hf_dataset.select_columns(meta_cols).to_pandas()

    # Build plot dataframe in the same order as tokenized_test_ds_to_plot
    plot_df = pd.DataFrame(coords, columns=["umap_1", "umap_2"])
    plot_df["sequence_id"] = umap_ds["sequence_id"]
    plot_df["segment_id"]  = umap_ds["segment_id"]
    plot_df["label_id"]    = umap_ds["labels"]

    # Join metadata (many-to-one should hold per segment_id; if not, switch to validate="many_to_many")
    plot_df = plot_df.merge(
        meta_df,
        on=["sequence_id", "segment_id"],
        how="left",
        validate="one_to_one",
    )

    plt.figure(figsize=(12, 7))
    ax = sns.scatterplot(
        data=plot_df,
        x="umap_1",
        y="umap_2",
        hue="taxon_short",
        s=28,          # larger dots
        alpha=0.85,
        linewidth=0,
        palette="tab20",
    )

    ax.set_title("UMAP of ProkBERT embeddings (colored by taxa)", pad=12)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")

    sns.move_legend(ax, "upper left", bbox_to_anchor=(1.02, 1), title="taxon_short")
    plt.tight_layout()
    plt.savefig(join(OUTPUT_PATH, "curricular_finetuning_umap_before.png"), dpi=300)
    plt.close()


    #train_batch_size = 64
    #eval_batch_size = 64
    #num_train_epochs = 0.5

    backbone_lr = 1e-5
    head_lr = 5e-4
    use_bf16 = True

    backbone_params = [p for n, p in model.named_parameters() if n.startswith("bert.")]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("bert.")]

    print(model)


    optimizer = AdamW(
        [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ],
    )
    reset_matmul_precision(tf32=False)

    training_args = TrainingArguments(
            output_dir='eskapee_example',
            report_to="none",
            logging_steps=20,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            num_train_epochs=num_train_epochs,
            bf16=use_bf16,
            torch_compile=True,
            torch_compile_mode ="max-autotune",
            ddp_backend='nccl'
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_ds,
        eval_dataset=tokenized_test_ds,
        data_collator=data_collator,
        #eval_strategy="steps",
        #compute_metrics=evaluate_embeddings,
        optimizers=(optimizer, None),
    )

    trainer.train()
    model.save_pretrained(OUTPUT_PATH)

    trained_model = ProkBertForCurricularClassification.from_pretrained(OUTPUT_PATH)
    emb, coords = compute_umap_for_dataset(
        model=trained_model,
        dataset=umap_ds,
        data_collator=data_collator,
        batch_size=eval_batch_size,
        seed=42,
    )
    score_before = evaluate_embeddings(emb, umap_ds["labels"])
    score_after = evaluate_embeddings(emb, umap_ds["labels"])
    print(f"Silhouette score after the training: {score_after:.4f}")

    #coords are aligned with umap_ds order
    plot_df = pd.DataFrame(coords, columns=["umap_1", "umap_2"])
    meta_cols = ["sequence_id", "segment_id", "assembly", "taxon", "taxon_short", "taxon_name"]
    meta_df = hf_dataset.select_columns(meta_cols).to_pandas()

    # Build plot dataframe in the same order as tokenized_test_ds_to_plot
    plot_df = pd.DataFrame(coords, columns=["umap_1", "umap_2"])
    plot_df["sequence_id"] = umap_ds["sequence_id"]
    plot_df["segment_id"]  = umap_ds["segment_id"]
    plot_df["label_id"]    = umap_ds["labels"]

    # Join metadata (many-to-one should hold per segment_id; if not, switch to validate="many_to_many")
    plot_df = plot_df.merge(
        meta_df,
        on=["sequence_id", "segment_id"],
        how="left",
        validate="one_to_one",
    )

    plt.figure(figsize=(12, 7))
    ax = sns.scatterplot(
        data=plot_df,
        x="umap_1",
        y="umap_2",
        hue="taxon_short",
        s=28,          # larger dots
        alpha=0.85,
        linewidth=0,
        palette="tab20",
    )

    ax.set_title("UMAP of ProkBERT embeddings (colored by taxa)", pad=12)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")

    sns.move_legend(ax, "upper left", bbox_to_anchor=(1.02, 1), title="taxon_short")
    plt.tight_layout()
    plt.savefig(join(OUTPUT_PATH, "curricular_finetuning_umap_after.png"), dpi=300)
    plt.close()




if __name__ == "__main__":
    main()

