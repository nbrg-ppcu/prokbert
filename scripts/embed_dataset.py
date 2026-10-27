#!/usr/bin/env python3

import os
import time
import math
import glob
import yaml
import argparse

import torch
import numpy as np
from tqdm import tqdm

from prokbert.tokenizer import LCATokenizer
from prokbert.sequence_dataset import SequenceDataset


__doc__ = """
Compute pooled embeddings of every contig with a ProkBERT model and save them for EmbeddingDataset.

Every contig is tokenized, split into windows of --seq-len tokens (the rest goes into one shorter
window, without padding), run through the model, and mean/sum pooled in blocks of --pooling-len tokens.
Each contig is first saved to its own emb_{sequence_id}.pt file; the concat step joins them, in
sequence_id order, into embeddings_forward.pt and seq_tokens.pt, and writes config.yaml.

The contigs can be split across several processes (e.g. one per GPU) with --shard-id and --num-shards.
Each shard only writes its own emb_*.pt files; run the concat step once, after all shards finished,
with --concat-only. With --num-shards 1 (the default) the concat step runs automatically at the end.

Example (one GPU):
python3 scripts/embed_dataset.py \
    --model neuralbioinfo/prokbert-mini-long \
    --input-dir /home/osvathm/prokbert/examples/outputs/toy_example/ \
    --output-dir /home/osvathm/prokbert/examples/outputs/toy_example \
    --seq-len 2000 \
    --pooling-strategy mean \
    --pooling-len 2 \
    --batch-size 4 \
    --kmer 6 \
    --shift 2

Example (8 shards, then concat):
python3 scripts/embed_dataset.py ... --shard-id 0 --num-shards 8    # ... up to --shard-id 7
python3 scripts/embed_dataset.py ... --concat-only

A szkript feltételezi, hogy egy contig --batch-size darab ablaka egyszerre befér a GPU memóriájába.


Slurm command:
srun --job-name=ncbi-db-test-create-embedding-with-mini-long \
     --nodes=1 \
     --nodelist=bolka \
     --ntasks=8 \
     --gpus-per-task=1 \
     --cpus-per-task=4 \
     --mem=200G \
     --time=7-00:00:00 \
     --partition=hi \
     --output=/home/osvathm/prokbert/logs-embeds/%x_%j_%t.out \
     --error=/home/osvathm/prokbert/logs-embeds/%x_%j_%t.err \
     bash -c 'python3 /home/osvathm/prokbert/scripts/embed_dataset.py \
        --model /home/datasets/experiments/experiment-bioinfo/models/prokbert-mini-long \
        --input-dir /home/datasets/experiments/experiment-bioinfo/fine-tuning/ncbi-test-sequence-dataset-for-madin-categorical-gram-stain \
        --output-dir /home/datasets/experiments/experiment-bioinfo/fine-tuning/ncbi-test-sequence-dataset-for-madin-categorical-gram-stain \
        --seq-len 2000 \
        --pooling-len 100 \
        --batch-size 32 \
        --kmer 6 \
        --shift 2 \
        --pooling-strategy mean \
        --shard-id $SLURM_PROCID \
        --num-shards $SLURM_NTASKS' \
        --verbose

Ha minden shard végzett, egyszer futtasd az összefűzést. Ehhez nem kell GPU:
srun --job-name=ncbi-db-test-concat-embeddings \
     --nodes=1 \
     --cpus-per-task=4 \
     --mem=150G \
     --partition=hi \
     --output=/home/osvathm/prokbert/logs-embeds/%x_%j_%t.out \
     --error=/home/osvathm/prokbert/logs-embeds/%x_%j_%t.err \
     python3 /home/osvathm/prokbert/scripts/embed_dataset.py \
        --model /home/datasets/experiments/experiment-bioinfo/models/prokbert-mini-long \
        --input-dir /home/datasets/experiments/experiment-bioinfo/fine-tuning/ncbi-test-sequence-dataset-for-madin-categorical-gram-stain \
        --output-dir /home/datasets/experiments/experiment-bioinfo/fine-tuning/ncbi-test-sequence-dataset-for-madin-categorical-gram-stain \
        --seq-len 2000 --pooling-len 100 --kmer 6 --shift 2 --pooling-strategy mean \
        --concat-only
"""


TOKENIZER = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--model", required=True, help="Hugging Face model name or local path.")
    parser.add_argument("--kmer", type=int, default=6, help="K-mer size for the tokenizer (default: same as --model).")
    parser.add_argument("--shift", type=int, default=2, help="Shift size for the tokenizer (default: same as --model).")

    parser.add_argument("--input-dir", required=True, help="Directory used for SequenceDataset to load data.")
    parser.add_argument("--output-dir", required=True, help="Directory for the embedding files.")
    parser.add_argument("--seq-len", type=int, default=2000, help="Window length in tokens for the model (default: 2000).")
    parser.add_argument("--pooling-strategy", default="mean", help="Pooling strategy for the embeddings (default: mean).")
    parser.add_argument("--pooling-len", type=int, default=100, help="Pooling length in tokens (default: 100).")
    parser.add_argument("--batch-size", type=int, default=64, help="Max. number of windows of one contig per forward pass.")
    parser.add_argument("--dtype", default="bfloat16", help="data type for model inference")

    parser.add_argument("--shard-id", type=int, default=0, help="Which shard this process handles (0-based).")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards (e.g. number of GPUs).")
    parser.add_argument("--concat-only", action="store_true", help="Only join the emb_*.pt files of all shards.")

    parser.add_argument("--verbose", action="store_true", help="enable verbose output.")
    args = parser.parse_args()

    if not 0 <= args.shard_id < args.num_shards:
        parser.error(f"--shard-id must be in [0, {args.num_shards}), got {args.shard_id}")
    if args.seq_len % args.pooling_len != 0:
        parser.error(f"--seq-len ({args.seq_len}) must be divisible by --pooling-len ({args.pooling_len})")
    return args


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
    # if args.verbose:
    #     print(f"Input ids batch of shape {batch['input_ids'].shape}")
    #     print(f"Attention mask shape: {batch['attention_mask'].shape}")
    with torch.autocast(device_type=DEVICE.type, dtype=model_dtype):
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask
        )
    last_hidden_state = outputs[0] # encoder output (not pooler output)
    if args.verbose:
        print(f"created main embeddings in shape {last_hidden_state.shape} in {time.perf_counter() - ti:.4f} seconds.")
    return last_hidden_state


def embed_in_chunks(batch, model: torch.nn.Module, args):
    """
    Run the windows of one contig through the model in chunks of --batch-size windows,
    so that a long contig (many windows) does not run out of GPU memory.

    Example:

    batch["input_ids"]:  (70, 2000)      <- 70 ablak, egyenként 2000 token

    chunks:
        0. csomag:  ablakok  0-31   →  (32, 2000)
        1. csomag:  ablakok 32-63   →  (32, 2000)
        2. csomag:  ablakok 64-69   →  ( 6, 2000)   <- az utolsó lehet kisebb

    embed minden csomagra:
    (32, 2000, 384), (32, 2000, 384), (6, 2000, 384)

    torch.cat(..., dim=0)  ->  (70, 2000, 384)    <- ugyanaz, mintha egyben futott volna
    """
    n_windows = batch["input_ids"].shape[0]
    chunks = [
        {key: value[i:i + args.batch_size] for key, value in batch.items()}
        for i in range(0, n_windows, args.batch_size)
    ]
    return torch.cat([embed(chunk, model, args) for chunk in chunks], dim=0)


def create_batches(tokenized_contig, args):

    seq_len = args.seq_len

    input_ids = tokenized_contig["input_ids"].reshape(-1)           # (n_tokens,)
    attention_mask = tokenized_contig["attention_mask"].reshape(-1) # (n_tokens,)

    n_windows = input_ids.numel() // seq_len    # number of full windows of seq_len tokens

    input_ids_main = input_ids[:n_windows * seq_len]
    attention_mask_main = attention_mask[:n_windows * seq_len]

    batch = None
    # Only build the main batch if the contig has at least one full window of seq_len tokens.
    # A shorter contig (e.g. 200 tokens with seq_len=2000) has no full window: the slice above is
    # empty and would become a (0, seq_len) tensor, which the model cannot process. In that case
    # all of the contig's tokens go into batch_rest instead.
    if n_windows > 0:
        batch = {
            "input_ids": input_ids_main.reshape(-1, seq_len), # (n_windows, seq_len)
            "attention_mask": attention_mask_main.reshape(-1, seq_len), # (n_windows, seq_len)
        }

    batch_rest = None
    if input_ids.numel() % seq_len != 0:
        input_ids_rest = input_ids[n_windows * seq_len:]
        attention_mask_rest = attention_mask[n_windows * seq_len:]

        batch_rest = {
            "input_ids": input_ids_rest.reshape(1, -1), # (1, n_tokens - n_windows * seq_len)
            "attention_mask": attention_mask_rest.reshape(1, -1), # (1, n_tokens - n_windows * seq_len)
        }

    return batch, batch_rest


def pooling(embeddings, pooling_strategy, dim=1):
    if pooling_strategy == "mean": embeddings = embeddings.mean(dim=dim)
    elif pooling_strategy == "sum": embeddings = embeddings.sum(dim=dim)
    else: raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")
    return embeddings


def pool_embeds(batch_embeds: torch.Tensor | None, batch_embeds_rest: torch.Tensor | None, args):
    pool = args.pooling_len
    dim = batch_embeds.shape[-1] if batch_embeds is not None else batch_embeds_rest.shape[-1]

    pooled = None
    if batch_embeds is not None:
        # main part: every window is a multiple of pooling_len, so it splits into full blocks
        pooled = pooling(batch_embeds.reshape(-1, pool, dim), args.pooling_strategy, dim=1)

    if batch_embeds_rest is not None:
        rest = batch_embeds_rest.reshape(-1, dim)                 # (n_rest_tokens, dim)
        # split into blocks of pooling_len; the last block may be shorter
        # (e.g. 1551 tokens with pooling_len=100 (seq_len=2000) -> 15 blocks of 100 + 1 block of 51 -> 16 rows)
        blocks = torch.split(rest, pool, dim=0)
        rest_pooled = torch.stack([pooling(b, args.pooling_strategy, dim=0) for b in blocks])
        pooled = torch.cat([pooled, rest_pooled], dim=0) if pooled is not None else rest_pooled

    if args.verbose: print(f"pooled batch final shape: {pooled.shape}")
    return pooled


def contig_path(args, sequence_id) -> str:
    return os.path.join(args.output_dir, f"emb_{sequence_id}.pt")


def process(sequence_dataset: SequenceDataset, model, tokenizer, args) -> None:

    os.makedirs(args.output_dir, exist_ok=True)

    t0 = time.perf_counter()
    print(f"Processing shard {args.shard_id}/{args.num_shards}")
    # this shard handles every num_shards-th contig, starting at shard_id
    contigs = sequence_dataset.metadata[args.shard_id::args.num_shards]

    for contig in tqdm(contigs, desc=f"shard {args.shard_id}/{args.num_shards}"):

        if args.verbose: print(f"\nprocessing {contig}")

        out_path = contig_path(args, contig["sequence_id"])

        # already processed (e.g. by an interrupted earlier run); the file also stores num_tokens,
        # so seq_tokens can be rebuilt from the files in the concat step
        if os.path.exists(out_path):
            continue

        start, end = sequence_dataset.get_coordinates_from_sequence_id(contig['sequence_id'])
        sequence = sequence_dataset.get_sequence_by_absolute_coordinates(start, end)

        if isinstance(sequence, np.ndarray):
            sequence = sequence.tobytes().decode("ascii")

        tokenized_contig = tokenizer(
            sequence, return_tensors="pt", padding=False,
            truncation=False, add_special_tokens=False
        )
        num_tokens = tokenized_contig["input_ids"].numel()
        if args.verbose: print(f"number of tokens: {num_tokens}")

        # egy adott s szegmens hosszal egy adott contig-ot összesen n * s darabra lehet felosztani,
        # de lehetséges hogy ezen túl kimarad néhány nukleotid. például ha a contig hossza 1050 és
        # a szegments hossz 100, akkor 1050 / 100 = 10 darab teljes szegmens és 50 nukleotid marad ki.
        # Balázs azt kérte, hogy a maradékra ne használjunk padding-et, majd pooling-ot, hanem közvetlenül
        # alkalmazzuk a megadott pooling-ot (pl. mean, sum), amit egy külön batch-be tudunk megtenni,
        # amit majd hozzáfőzünk a fő batch-hez.
        if num_tokens == 0:
            # contig shorter than one k-mer: no tokens, no rows
            batch_embeds = torch.empty(0, model.config.hidden_size)
        else:
            batch, batch_rest = create_batches(tokenized_contig, args)
            batch_embeds = embed_in_chunks(batch, model, args) if batch is not None else None # (n_windows, seq_len, dim)
            batch_embeds_rest = embed(batch_rest, model, args) if batch_rest is not None else None # (1, n_rest_tokens, dim)
            batch_embeds = pool_embeds(batch_embeds, batch_embeds_rest, args)

        expected_rows = math.ceil(num_tokens / args.pooling_len)
        assert batch_embeds.shape[0] == expected_rows, \
            f"sequence_id {contig['sequence_id']}: {batch_embeds.shape[0]} rows, expected {expected_rows}"

        torch.save(
            {"start": start, "num_tokens": num_tokens, "embeddings": batch_embeds.detach().cpu().clone()},
            out_path,
        )

    if args.verbose: print(f"Shard {args.shard_id} finished in {time.perf_counter() - t0:.4f} seconds.")


def concat_embeddings(sequence_dataset: SequenceDataset, args) -> None:
    """Join the emb_*.pt files of all shards, in sequence_id order, and rebuild seq_tokens from them."""
    files = glob.glob(os.path.join(args.output_dir, "emb_*.pt"))
    # sort numerically by sequence_id, not alphabetically ("emb_10" must come after "emb_2")
    files.sort(key=lambda f: int(os.path.basename(f)[len("emb_"):-len(".pt")]))

    # every contig must have its file, otherwise a shard has not finished (or failed)
    n_contigs = len(sequence_dataset.metadata)
    if len(files) != n_contigs:
        raise RuntimeError(
            f"Found {len(files)} emb_*.pt files for {n_contigs} contigs. "
            f"Did every shard finish? Run the missing shards again before --concat-only."
        )

    embeddings, seq_tokens = [], []
    for f in tqdm(files, desc="concat"):
        data = torch.load(f, weights_only=True)
        embeddings.append(data["embeddings"])
        seq_tokens.append(data["num_tokens"])

    all_emb = torch.cat(embeddings, dim=0)
    if args.verbose: print(f"final embeddings shape: {all_emb.shape}")

    torch.save(all_emb, os.path.join(args.output_dir, "embeddings_forward.pt"))
    torch.save(torch.tensor(seq_tokens), os.path.join(args.output_dir, "seq_tokens.pt"))
    for f in files: os.remove(f)


def save_config(args) -> None:
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


def main() -> None:

    args = parse_args()

    sequence_dataset = SequenceDataset()
    sequence_dataset.load_dataset(args.input_dir, to_string=False)

    if not args.concat_only:
        model = load_model(args)
        tokenizer = get_tokenizer(args)
        process(sequence_dataset, model, tokenizer, args)

    # with several shards, only concat when explicitly asked: the other shards may still be running
    if args.concat_only or args.num_shards == 1:
        concat_embeddings(sequence_dataset, args)
        save_config(args)


if __name__ == "__main__":
    main()


