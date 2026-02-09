from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from prokbert.sequtils import *
from prokbert.training_utils import *
from prokbert.models2 import ProkBertForCurricularClassification
from prokbert.tokenizer import LCATokenizer
from prokbert.curriculum_utils import compute_umap_for_dataset, evaluate_embeddings
from prokbert.curriculum_custom_trainier import CustomTrainer
from datasets import Dataset, load_dataset, ClassLabel
from transformers import EarlyStoppingCallback
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
from os.path import join
import os

REPO_ID = "neuralbioinfo/eskapee"
MODEL_NAME = "neuralbioinfo/mini2-c"
OUTPUT_PATH = "./"

if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemError('GPU device not found')
    else:
        device_name = torch.cuda.get_device_name(0)
        print(f'Found GPU at: {device_name}')
    num_cores = os.cpu_count()
    print(f'Number of available CPU cores: {num_cores}')

    #Loading dataset
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

    sequences = ds.to_pandas()
    max_length = 512
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

    tokenizer = LCATokenizer(kmer=1, shift=1)
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

    curricular_face_m = 0.5          # angular margin (typical range ~0.2–0.6)
    curricular_face_s = 64.0         # logit scale (typical range ~16–64)
    classification_dropout_rate = 0.1
    curriculum_hidden_size = 128     # embedding/projection size used by the curricular head (often 128)

    use_bf16 = True
    model_dtype = torch.bfloat16 if use_bf16 else torch.float32


    model = ProkBertForCurricularClassification.from_pretrained(
        MODEL_NAME,
        curricular_num_labels=len(id2label),
        curricular_face_m=curricular_face_m,
        curricular_face_s=curricular_face_s,
        classification_dropout_rate=classification_dropout_rate,
        curriculum_hidden_size=curriculum_hidden_size,
        torch_dtype=model_dtype,
        id2label=id2label,
        label2id=label2id,
        reference_compile=False,
    )

    umap_n = 2000
    umap_seed = 123

    umap_ds = tokenized_test_ds.shuffle(seed=umap_seed)#.select(range(min(umap_n, len(tokenized_train_ds))))
    emb, coords = compute_umap_for_dataset(
        model=model,
        dataset=umap_ds,
        data_collator=data_collator,
        batch_size=128,
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
    plt.savefig(join(OUTPUT_PATH, "curricular_finetuning_1_5_umap_before.png"), dpi=300)
    plt.close()


    train_batch_size = 64
    eval_batch_size = 64
    num_train_epochs = 0.5
    num_eval_steps = 40
    use_bf16 = False

    training_args = TrainingArguments(
            output_dir=join(OUTPUT_PATH, "eskapee_example"),
            overwrite_output_dir=True,
            report_to="none",
            logging_steps=20,
            evaluation_strategy="steps",
            eval_steps = num_eval_steps,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            num_train_epochs=num_train_epochs,
            metric_for_best_model="eval_silhouette_score",
            greater_is_better=True,
            bf16=use_bf16,
            torch_compile=False,
        )

    training_args.backbone_lr_rate = 1e-5
    training_args.head_lr_rate = 5e-4
    training_args.beta_1 = 0.5794
    training_args.beta_2 = 0.6576

    trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_ds,
            eval_dataset=tokenized_test_ds,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

    trainer.train()

    trainer.save_model(join(OUTPUT_PATH, "curricular_finetuning_1_5_final_model"))

    model = ProkBertForCurricularClassification.from_pretrained(
        join(OUTPUT_PATH, "curricular_finetuning_1_5_final_model"),
        torch_dtype=model_dtype,)

    emb, coords = compute_umap_for_dataset(
        model=model,
        dataset=umap_ds,
        data_collator=data_collator,
        batch_size=128,
        seed=42,
    )
    score_after = evaluate_embeddings(emb, umap_ds["labels"])
    print(f"Silhouette score after training: {score_after:.4f}")

    # coords are aligned with umap_ds order
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
    plt.show()

    plt.savefig(join(OUTPUT_PATH, "curricular_finetuning_1_5_umap_after.png"), dpi=300) 







    


