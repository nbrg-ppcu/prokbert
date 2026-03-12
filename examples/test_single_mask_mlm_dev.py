from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import torch
from transformers import set_seed
from transformers.utils import logging as hf_logging

from prokbert.models import ProkBertForMaskedLM
from prokbert.tokenizer import LCATokenizer


@dataclass
class Config:
    model_ids: tuple[str, ...] = (
        "neuralbioinfo/prokbert-mini",
        "neuralbioinfo/prokbert-mini-c",
        "neuralbioinfo/prokbert-mini-long",
    )
    sequences: tuple[str, ...] = (
        "ATGTCCGCGGGACCTAACGATCGATCGTACGATCGATCGTACGATCGATCGATGCTAGCTAGCTAGCATCG",
        "TTGACATTTGCCGTTAACCGGATTTGCGATCGTAGCTAGGCTAACCGTTAAGGCTTACCGATTAACCGGTA",
    )
    output_dir: str = "./mlm_one_mask_results"
    seed: int = 42
    top_k: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def tokenize_one(tokenizer, sequence: str, max_length: int | None):
    kwargs = {
        "return_tensors": "pt",
        "truncation": True,
    }
    if max_length is not None:
        kwargs["max_length"] = max_length

    enc = tokenizer(sequence, **kwargs)

    if "token_type_ids" not in enc:
        enc["token_type_ids"] = torch.zeros_like(enc["input_ids"])

    return enc


def build_special_mask_from_input_ids(input_ids_1d: torch.Tensor, tokenizer) -> torch.Tensor:
    special_mask = torch.zeros_like(input_ids_1d, dtype=torch.bool)
    for sid in tokenizer.all_special_ids:
        special_mask |= input_ids_1d.eq(int(sid))
    return special_mask


def choose_mask_position(enc, tokenizer) -> int:
    input_ids_1d = enc["input_ids"][0]
    attention_mask_1d = enc["attention_mask"][0].bool()
    special_mask_1d = build_special_mask_from_input_ids(input_ids_1d, tokenizer)

    valid_positions = torch.where(attention_mask_1d & ~special_mask_1d)[0].tolist()
    valid_positions = [int(p) for p in valid_positions]

    if not valid_positions:
        raise ValueError("No valid non-special token positions found.")

    # Avoid first/last real token when possible.
    core_positions = valid_positions[1:-1]
    if core_positions:
        valid_positions = core_positions

    # Deterministic: choose the middle valid token.
    return valid_positions[len(valid_positions) // 2]


def make_masked_window(tokenizer, input_ids_1d: torch.Tensor, pos: int, radius: int = 4) -> str:
    start = max(0, pos - radius)
    end = min(input_ids_1d.shape[0], pos + radius + 1)

    token_ids = input_ids_1d[start:end].tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    tokens[pos - start] = tokenizer.mask_token
    return " ".join(tokens)


def evaluate_model(model_id: str, cfg: Config):
    print(f"\n=== Evaluating {model_id} ===")

    tokenizer = LCATokenizer.from_pretrained(model_id)
    model = ProkBertForMaskedLM.from_pretrained(model_id).to(cfg.device).eval()

    max_length = getattr(model.config, "max_position_embeddings", None)

    results = []

    for seq_idx, sequence in enumerate(cfg.sequences):
        enc = tokenize_one(tokenizer, sequence, max_length=max_length)

        input_ids_1d = enc["input_ids"][0]
        attention_mask_1d = enc["attention_mask"][0]

        print(
            f"seq={seq_idx} encoded_len={input_ids_1d.numel()} "
            f"real_tokens={int(attention_mask_1d.sum().item())}"
        )

        pos = choose_mask_position(enc, tokenizer)

        original_id = int(input_ids_1d[pos].item())
        original_token = tokenizer.convert_ids_to_tokens([original_id])[0]

        masked_batch = {k: v.clone().to(cfg.device) for k, v in enc.items()}
        masked_batch["input_ids"][0, pos] = tokenizer.mask_token_id

        with torch.inference_mode():
            outputs = model(**masked_batch, return_dict=True)
            logits = outputs.logits[0, pos].detach().cpu()

        probs = torch.softmax(logits, dim=-1)

        original_rank = int((logits > logits[original_id]).sum().item() + 1)

        k = min(cfg.top_k, logits.shape[-1])
        topk = torch.topk(logits, k=k)
        topk_ids = [int(x) for x in topk.indices.tolist()]
        topk_tokens = tokenizer.convert_ids_to_tokens(topk_ids)
        topk_probs = [float(probs[i].item()) for i in topk_ids]

        result = {
            "model_id": model_id,
            "sequence_index": seq_idx,
            "mask_position": int(pos),
            "original_token_id": original_id,
            "original_token": original_token,
            "original_rank": original_rank,
            "pass_top10": bool(original_rank <= cfg.top_k),
            "target_prob": float(probs[original_id].item()),
            "topk_ids": topk_ids,
            "topk_tokens": topk_tokens,
            "topk_probs": topk_probs,
            "masked_window": make_masked_window(tokenizer, input_ids_1d, pos),
            "raw_sequence": sequence,
        }
        results.append(result)

        print(
            f"seq={seq_idx} pos={pos} token={original_token} "
            f"rank={original_rank} top10={original_rank <= cfg.top_k}"
        )
        print("top10:", topk_tokens)

    return results


def main():
    cfg = Config()

    hf_logging.set_verbosity_error()
    set_seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    out_root = Path(cfg.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    failures = []

    for model_id in cfg.model_ids:
        results = evaluate_model(model_id, cfg)

        model_dir = out_root / model_id.split("/")[-1]
        model_dir.mkdir(parents=True, exist_ok=True)

        with open(model_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        n_pass = sum(int(r["pass_top10"]) for r in results)
        n_total = len(results)
        print(f"{model_id}: {n_pass}/{n_total} passed top{cfg.top_k}")

        if n_pass != n_total:
            failures.append(model_id)

    if failures:
        raise SystemExit(
            "Single-mask MLM test failed for: " + ", ".join(failures)
        )

    print("\nSingle-mask MLM test passed for all models.")


if __name__ == "__main__":
    main()