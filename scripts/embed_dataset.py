#!/usr/bin/env python3

import math
from typing import Any

import os
import time
import glob
import yaml
import argparse

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from prokbert.tokenizer import LCATokenizer
from prokbert.sequence_dataset import SequenceDataset


__doc__ = """
Compute embeddings with a ProkBERT models and save them to one .npy file
for EmbeddingDataset to use later.

The input is split into contiguous segments (every base covered once). Row i of the
output array is the embedding of the segment with segment_id i, so the order does not
depend on the number of DataLoader workers. Segment metadata (contig, coordinates) is
written next to the .npy as a .csv with the same row order.

Example:
python3 scripts/embed_dataset.py \
    --model neuralbioinfo/prokbert-mini-long \
    --input-dir /home/osvathm/prokbert/examples/outputs/toy_example/ \
    --output-dir /home/osvathm/prokbert/examples/outputs/toy_example \
    --seq-len 2000 \
    --pooling-strategy mean \
    --pooling-len 2 \
    --batch-size 4 \
    --kmer 6 \
    --shift 2 \
    --verbose

A szkript feltételezi, hogy bármely contig hosszából alkotott tokenek teljesen beleférnek
a GPU memóriájába, ami az A100 80 GB VRAM-on teljesül. Ha ez nem igaz, akkor a szkript elhal.
"""


TOKENIZER = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--model", required=True, help="Hugging Face model name or local path.")
    parser.add_argument("--kmer", type=int, default=6, help="K-mer size for the tokenizer (default: same as --model).")
    parser.add_argument("--shift", type=int, default=2, help="Shift size for the tokenizer (default: same as --model).")

    parser.add_argument("--input-dir", required=True, help="Directory used for SequenceDataset to load data.")
    parser.add_argument("--output-dir", required=True, help="Output .npy path.")
    parser.add_argument("--seq-len", type=int, default=2000, help="Sequence length for the model (default: 2000).")
    parser.add_argument("--pooling-strategy", default="mean", help="Pooling strategy for the embeddings (default: mean).")
    parser.add_argument("--pooling-len", type=int, default=100, help="Pooling length for the embeddings (default: 100).")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dtype", default="bfloat16", help="data type for model inference")
    parser.add_argument("--verbose", action="store_true", help="enable verbose output.")
    return parser.parse_args()


def get_tokenizer(args) -> LCATokenizer:
    global TOKENIZER
    if TOKENIZER is None:
        if args.verbose: print(f"Building tokenizer in process with kmer={args.kmer}, shift={args.shift}")
        TOKENIZER = LCATokenizer(kmer=args.kmer, shift=args.shift)
    return TOKENIZER


def load_model(args):

    if "prokbert-mini-long" in args.model:
        from prokbert.models.megatronbert import ProkBertForMaskedLM
    elif "mini2-c" in args.model:
        from prokbert.models.modernbert import ProkBertForMaskedLM
    else:
        raise ValueError(f"Unknown base model: {args.model}")

    prokbert_model = ProkBertForMaskedLM.from_pretrained(args.model)
    # masked lm has an additional head, we need the base model for embeddings
    # (naming convention differs between models)
    model = prokbert_model.model if hasattr(prokbert_model, "model") else prokbert_model.bert

    model = model.to(DEVICE)
    return model


@torch.inference_mode()
def embed(batch, model: torch.nn.Module, args):
    ti = time.perf_counter()
    model_dtype = getattr(torch, args.dtype)
    input_ids = batch["input_ids"].to(DEVICE)
    attention_mask = batch["attention_mask"].to(DEVICE)
    with torch.autocast(device_type=DEVICE.type, dtype=model_dtype):
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask
        )
    last_hidden_state = outputs[0] # encoder output (not pooler output)
    if args.verbose:
        print(f"created main embeddings in shape {last_hidden_state.shape} in {time.perf_counter() - ti:.4f} seconds.")
    return last_hidden_state


def create_batches(tokenized_contig, args, pad_id=0):

    seq_len = args.seq_len
    pooling_len = args.pooling_len

    assert seq_len % pooling_len == 0, "sequence length must be divisible by pooling length"

    input_ids = tokenized_contig["input_ids"].reshape(-1)           # (n_tokens,)
    attention_mask = tokenized_contig["attention_mask"].reshape(-1) # (n_tokens,)

    num_tokens = input_ids.numel() // seq_len

    input_ids_main = input_ids[:num_tokens * seq_len]
    attention_mask_main = attention_mask[:num_tokens * seq_len]

    batch = {
        "input_ids": input_ids_main.reshape(-1, seq_len), # (n_batches, seq_len)
        "attention_mask": attention_mask_main.reshape(-1, seq_len), # (n_batches, seq_len)
    }

    batch_rest = None
    if input_ids.numel() % seq_len != 0:
        input_ids_rest = input_ids[num_tokens * seq_len:]
        attention_mask_rest = attention_mask[num_tokens * seq_len:]

        batch_rest = {
            "input_ids": input_ids_rest.reshape(1, -1), # (1, input_ids.numel() - num_tokens * seq_len)
            "attention_mask": attention_mask_rest.reshape(1, -1), # (1, input_ids.numel() - num_tokens * seq_len)
        }

    return batch, batch_rest


def pooling(embeddings, pooling_strategy, dim=1):
    if pooling_strategy == "mean": embeddings = embeddings.mean(dim=dim)
    elif pooling_strategy == "sum": embeddings = embeddings.sum(dim=dim)
    else: raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")
    return embeddings


def pool_embeds(batch_embeds: torch.Tensor, batch_embeds_rest: torch.Tensor | None, args):
    pool = args.pooling_len
    dim = batch_embeds.shape[-1]

    # main part: every window is a multiple of pooling_len, so it splits into full blocks
    pooled = pooling(batch_embeds.reshape(-1, pool, dim), args.pooling_strategy, dim=1)

    if batch_embeds_rest is not None:
        rest = batch_embeds_rest.reshape(-1, dim)                 # (n_rest_tokens, dim)
        # split into blocks of pooling_len; the last block may be shorter
        # (e.g. 1551 tokens with pooling_len=100 (seq_len=2000) -> 15 blocks of 100 + 1 block of 51 -> 16 rows)
        blocks = torch.split(rest, pool, dim=0)
        rest_pooled = torch.stack([pooling(b, args.pooling_strategy, dim=0) for b in blocks])
        pooled = torch.cat([pooled, rest_pooled], dim=0)

    if args.verbose: print(f"pooled batch final shape: {pooled.shape}")
    return pooled


def process(sequence_dataset: SequenceDataset, model, tokenizer, args) -> None:

    os.makedirs(args.output_dir, exist_ok=True)

    t0 = time.perf_counter()
    metadata = sequence_dataset.metadata
    seq_tokens = []

    for contig in tqdm(metadata):

        if args.verbose: print(f"\nprocessing {contig}")

        out_path = os.path.join(args.output_dir, f"emb_{contig['sequence_id']}.pt")

        if os.path.exists(out_path):
            continue

        start, end = sequence_dataset.get_coordinates_from_sequence_id(contig['sequence_id'])
        sequence = sequence_dataset.get_sequence_by_absolute_coordinates(start, end)

        if isinstance(sequence, np.ndarray):
            sequence = sequence.tobytes().decode("ascii")

        if args.verbose: print(f"sequence: {sequence}")

        tokenized_contig = tokenizer( # (1, seq_len)
            sequence, return_tensors="pt", padding=False,
            truncation=False, add_special_tokens=False
        )
        num_tokens = tokenized_contig["input_ids"].numel()
        if args.verbose: print(f"number of tokens: {num_tokens}")
        seq_tokens.append(num_tokens)

        # egy adott s szegmens hosszal egy adott contig-ot összesen n * s darabra lehet felosztani,
        # de lehetséges hogy ezen túl kimarad néhány nukleotid. például ha a contig hossza 1050 és
        # a szegments hossz 100, akkor 1050 / 100 = 10 darab teljes szegmens és 50 nukleotid marad ki.
        # Balázs azt kérte, hogy a maradékra ne használjunk padding-et, majd pooling-ot, hanem közvetlenül
        # alkalmazzuk a megadott pooling-ot (pl. mean, sum), amit egy külön batch-be tudunk megtenni,
        # amit majd hozzáfőzünk a fő batch-hez.
        batch, batch_rest = create_batches(tokenized_contig, args)

        batch_embeds = embed(batch, model, args) # (n_batches, seq_len)
        batch_embeds_rest = embed(batch_rest, model, args) if batch_rest is not None else None # (1, m)

        batch_embeds = pool_embeds(batch_embeds, batch_embeds_rest, args)

        expected_rows = math.ceil(num_tokens / args.pooling_len)
        assert batch_embeds.shape[0] == expected_rows, \
            f"sequence_id {contig['sequence_id']}: {batch_embeds.shape[0]} rows, expected {expected_rows}"

        torch.save({"start": start, "embeddings": batch_embeds.detach().cpu().clone()}, out_path)

    torch.save(torch.tensor(seq_tokens), os.path.join(args.output_dir, "seq_tokens.pt"))

    if args.verbose: print(f"Finished processing in {time.perf_counter() - t0:.4f} seconds.")


def concat_embeddings(args) -> None:
    files = glob.glob(os.path.join(args.output_dir, "emb_*.pt"))
    # sort numerically by sequence_id, not alphabetically ("emb_10" must come after "emb_2")
    files.sort(key=lambda f: int(os.path.basename(f)[len("emb_"):-len(".pt")]))

    all_emb = torch.cat([torch.load(f, weights_only=True)["embeddings"] for f in files], dim=0)
    if args.verbose: print(f"final embeddings shape: {all_emb.shape}")
    torch.save(all_emb, os.path.join(args.output_dir, "embeddings_forward.pt"))
    for f in files: os.remove(f)

def main() -> None:

    args = parse_args()

    sequence_dataset = SequenceDataset()
    sequence_dataset.load_dataset(args.input_dir, to_string=False)

    model = load_model(args)
    tokenizer = get_tokenizer(args)

    process(sequence_dataset, model, tokenizer, args)
    concat_embeddings(args)

    config = {
        "tokenizer": "LCATokenizer",
        "kmer": args.kmer,
        "shift": args.shift,
        "model": args.model,
        "pooling_strategy": args.pooling_strategy,
        "pooling_length": args.pooling_len,
        "special_tokens": []
    }
    with open(save_path := os.path.join(args.output_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)
    if args.verbose: print(f"Saved config to {save_path}")


if __name__ == "__main__":
    main()

