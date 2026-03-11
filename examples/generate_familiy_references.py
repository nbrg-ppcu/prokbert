from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, set_seed

from prokbert.models import ProkBertModel


MODEL_IDS = [
    "neuralbioinfo/prokbert-mini",
    "neuralbioinfo/prokbert-mini-c",
    "neuralbioinfo/prokbert-mini-long",
]

SEQUENCES = [
    "ATGTCCGCGGG",
    "TTGACATTT",
]

OUT_ROOT = Path("../data/test_data")
DEVICE = torch.device("cpu")


def masked_mean(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1)
    return summed / denom


def main() -> None:
    set_seed(42)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for model_id in MODEL_IDS:
        repo_name = model_id.split("/")[-1]
        out_dir = OUT_ROOT / repo_name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(out_dir)

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = ProkBertModel.from_pretrained(model_id).to(DEVICE).eval()

        batch = tokenizer(
            SEQUENCES,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        if "token_type_ids" not in batch:
            batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])

        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        with torch.inference_mode():
            outputs = model(**batch, return_dict=True)

        last_hidden_state = outputs.last_hidden_state.cpu()
        attention_mask = batch["attention_mask"].cpu()

        payload = {
            "model_id": model_id,
            "sequences": SEQUENCES,
            "input_ids": batch["input_ids"].cpu(),
            "attention_mask": attention_mask,
            "token_type_ids": batch["token_type_ids"].cpu(),
            "last_hidden_state": last_hidden_state,
            "masked_mean": masked_mean(last_hidden_state, attention_mask),
            "cls": last_hidden_state[:, 0, :],
        }

        torch.save(payload, out_dir / "base_reference.pt")

        meta = {
            "model_id": model_id,
            "n_sequences": len(SEQUENCES),
            "sequence_lengths": [len(s) for s in SEQUENCES],
            "tensor_file": "base_reference.pt",
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

        print(f"saved {model_id} -> {out_dir / 'base_reference.pt'}")


if __name__ == "__main__":
    main()