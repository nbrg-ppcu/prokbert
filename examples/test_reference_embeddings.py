from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from transformers import AutoTokenizer, set_seed
from transformers.utils import logging as hf_logging

from prokbert.models import (
    ProkBertModel,
    ProkBertForMaskedLM,
    # Uncomment later if you want to test these too and the source repo actually contains such weights
    # ProkBertForSequenceClassification,
    # ProkBertForSequenceClassificationExt,
    # ProkBertForCurricularClassification,
)


@dataclass
class Config:
    # Key = reference folder name under ../data/test_data
    # Value = model source to test (HF repo id or local folder)
    model_sources: dict[str, str] = None

    # Which model wrappers to test against the same base reference
    # "base" compares ProkBertModel directly
    # "mlm" compares ProkBertForMaskedLM(...).bert
    model_variants: tuple[str, ...] = ("base",)

    reference_root: Path = (Path(__file__).resolve().parent / "../data/test_data").resolve()
    device: torch.device = torch.device("cpu")

    # Tight default tolerances for CPU regression
    atol: float = 1e-6
    rtol: float = 1e-5
    seed: int = 42


def make_default_config() -> Config:
    return Config(
        model_sources={
            "prokbert-mini": "neuralbioinfo/prokbert-mini",
            "prokbert-mini-c": "neuralbioinfo/prokbert-mini-c",
            "prokbert-mini-long": "neuralbioinfo/prokbert-mini-long",
        },
        model_variants=("base",),
    )


def masked_mean(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1)
    return summed / denom


def load_reference(reference_root: Path, repo_name: str) -> dict:
    ref_path = reference_root / repo_name / "base_reference.pt"
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference file not found: {ref_path}")
    return torch.load(ref_path, map_location="cpu")


def get_variant_loaders() -> dict[str, Callable[[str], torch.nn.Module]]:
    return {
        "base": lambda source: ProkBertModel.from_pretrained(source),
        "mlm": lambda source: ProkBertForMaskedLM.from_pretrained(source),

        # Add these later if the tested source actually contains those task weights:
        # "seqcls": lambda source: ProkBertForSequenceClassification.from_pretrained(source),
        # "seqcls_ext": lambda source: ProkBertForSequenceClassificationExt.from_pretrained(source),
        # "curricular": lambda source: ProkBertForCurricularClassification.from_pretrained(source),
    }


def extract_base_encoder(model: torch.nn.Module) -> torch.nn.Module:
    # Base encoder repo
    if isinstance(model, ProkBertModel):
        return model

    # MLM / classifier wrappers
    if hasattr(model, "bert") and isinstance(model.bert, torch.nn.Module):
        return model.bert

    # Fallback: use the model itself
    return model


def prepare_batch(source: str, sequences: list[str] | tuple[str, ...], device: torch.device) -> dict[str, torch.Tensor]:
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)

    batch = tokenizer(
        list(sequences),
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    if "token_type_ids" not in batch:
        batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])

    batch = {k: v.to(device) for k, v in batch.items()}
    return batch


def compare_exact_tensor(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{name}: shape mismatch\n"
            f"  actual   = {tuple(actual.shape)}\n"
            f"  expected = {tuple(expected.shape)}"
        )

    if not torch.equal(actual.cpu(), expected.cpu()):
        raise AssertionError(
            f"{name}: exact mismatch\n"
            f"  actual:\n{actual.cpu()}\n"
            f"  expected:\n{expected.cpu()}"
        )


def compare_close_tensor(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float,
    rtol: float,
) -> None:
    actual_cpu = actual.detach().cpu()
    expected_cpu = expected.detach().cpu()

    if actual_cpu.shape != expected_cpu.shape:
        raise AssertionError(
            f"{name}: shape mismatch\n"
            f"  actual   = {tuple(actual_cpu.shape)}\n"
            f"  expected = {tuple(expected_cpu.shape)}"
        )

    if not torch.allclose(actual_cpu, expected_cpu, atol=atol, rtol=rtol):
        diff = (actual_cpu - expected_cpu).abs()
        max_abs = float(diff.max().item())
        mean_abs = float(diff.mean().item())

        # Relative diff guarded against zeros
        denom = expected_cpu.abs().clamp(min=1e-12)
        rel = diff / denom
        max_rel = float(rel.max().item())
        mean_rel = float(rel.mean().item())

        raise AssertionError(
            f"{name}: values differ\n"
            f"  atol={atol}, rtol={rtol}\n"
            f"  max_abs_diff={max_abs:.6e}\n"
            f"  mean_abs_diff={mean_abs:.6e}\n"
            f"  max_rel_diff={max_rel:.6e}\n"
            f"  mean_rel_diff={mean_rel:.6e}"
        )


def run_one_test(
    repo_name: str,
    source: str,
    variant: str,
    cfg: Config,
) -> None:
    print(f"\n=== Testing repo={repo_name}  source={source}  variant={variant} ===")

    reference = load_reference(cfg.reference_root, repo_name)

    batch = prepare_batch(source, reference["sequences"], cfg.device)

    # 1) Tokenization must match exactly
    compare_exact_tensor("input_ids", batch["input_ids"], reference["input_ids"])
    compare_exact_tensor("attention_mask", batch["attention_mask"], reference["attention_mask"])
    compare_exact_tensor("token_type_ids", batch["token_type_ids"], reference["token_type_ids"])

    # 2) Load the requested model wrapper and extract the base encoder
    loaders = get_variant_loaders()
    if variant not in loaders:
        raise KeyError(f"Unknown model variant: {variant}")

    full_model = loaders[variant](source).to(cfg.device).eval()
    base_encoder = extract_base_encoder(full_model).to(cfg.device).eval()

    # 3) Forward pass through the base encoder
    with torch.inference_mode():
        outputs = base_encoder(**batch, return_dict=True)

    actual_last_hidden = outputs.last_hidden_state.cpu()
    actual_attention_mask = batch["attention_mask"].cpu()
    actual_masked_mean = masked_mean(actual_last_hidden, actual_attention_mask)
    actual_cls = actual_last_hidden[:, 0, :]

    # 4) Compare against the saved reference
    compare_close_tensor(
        "last_hidden_state",
        actual_last_hidden,
        reference["last_hidden_state"],
        atol=cfg.atol,
        rtol=cfg.rtol,
    )
    compare_close_tensor(
        "masked_mean",
        actual_masked_mean,
        reference["masked_mean"],
        atol=cfg.atol,
        rtol=cfg.rtol,
    )
    compare_close_tensor(
        "cls",
        actual_cls,
        reference["cls"],
        atol=cfg.atol,
        rtol=cfg.rtol,
    )

    print("PASS")


def main() -> None:
    cfg = make_default_config()

    hf_logging.set_verbosity_error()
    set_seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    failures: list[str] = []

    print(f"Reference root: {cfg.reference_root}")
    print(f"Device: {cfg.device}")
    print(f"Variants: {cfg.model_variants}")

    for repo_name, source in cfg.model_sources.items():
        for variant in cfg.model_variants:
            try:
                run_one_test(repo_name, source, variant, cfg)
            except Exception as exc:
                failures.append(f"{repo_name} [{variant}] -> {type(exc).__name__}: {exc}")
                print(f"FAIL: {repo_name} [{variant}]")

    if failures:
        print("\nFailures:")
        for item in failures:
            print(f"  - {item}")
        raise SystemExit(1)

    print("\nAll embedding reference tests passed.")


if __name__ == "__main__":
    main()