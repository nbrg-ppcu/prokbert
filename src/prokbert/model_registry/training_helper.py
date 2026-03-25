from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Union

import pandas as pd
import torch
from huggingface_hub import hf_hub_download


DEFAULT_REGISTRY_REPO_ID = os.getenv("PROKBERT_TRAINING_REGISTRY_REPO", "neuralbioinfo/model-registry")
DEFAULT_REGISTRY_REVISION = os.getenv("PROKBERT_TRAINING_REGISTRY_REVISION", "main")

_BASEMODELS_FILENAME = "basemodels.json"
_DEFAULTS_FILENAME = "default_training_parameters.json"


def _load_records_from_hf_json(
    *,
    repo_id: str,
    filename: str,
    revision: str = DEFAULT_REGISTRY_REVISION,
    token: Optional[str] = None,
    local_files_only: bool = False,
) -> list[dict[str, Any]]:
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        revision=revision,
        token=token,
        local_files_only=local_files_only,
    )

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict) and isinstance(payload.get("records"), list):
        records = payload["records"]
    else:
        raise ValueError(
            f"Unsupported JSON structure in {filename!r}. Expected either a list of records "
            "or an object with a top-level 'records' array."
        )

    return records


def _load_registry_from_hf_dataset(
    *,
    repo_id: str,
    revision: str = DEFAULT_REGISTRY_REVISION,
    token: Optional[str] = None,
    local_files_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_records = _load_records_from_hf_json(
        repo_id=repo_id,
        filename=_BASEMODELS_FILENAME,
        revision=revision,
        token=token,
        local_files_only=local_files_only,
    )
    default_records = _load_records_from_hf_json(
        repo_id=repo_id,
        filename=_DEFAULTS_FILENAME,
        revision=revision,
        token=token,
        local_files_only=local_files_only,
    )

    model_db = pd.DataFrame(model_records)
    training_defaults = pd.DataFrame(default_records)
    return _normalize_registry_tables(model_db, training_defaults)


def _normalize_registry_tables(
    model_db: pd.DataFrame,
    training_defaults: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_db = model_db.copy()
    training_defaults = training_defaults.copy()

    if "model_id" not in model_db.columns:
        if "name" in model_db.columns:
            model_db["model_id"] = model_db["name"].astype(str)
        elif "hf_name" in model_db.columns:
            model_db["model_id"] = model_db["hf_name"].astype(str)
        else:
            raise ValueError("basemodels.json must include either 'model_id', 'name', or 'hf_name'.")

    if "name" not in model_db.columns:
        model_db["name"] = model_db["model_id"].astype(str)

    if "hf_path" not in model_db.columns and "hf_repo_id" in model_db.columns:
        model_db["hf_path"] = model_db["hf_repo_id"]
    if "hf_repo_id" not in model_db.columns and "hf_path" in model_db.columns:
        model_db["hf_repo_id"] = model_db["hf_path"]

    model_db["model_id"] = model_db["model_id"].astype(str)

    lookup: dict[str, str] = {}
    for _, row in model_db.iterrows():
        model_id = str(row["model_id"])
        for key in ("model_id", "name", "hf_name", "hf_path", "hf_repo_id"):
            if key in model_db.columns:
                value = row.get(key)
                if pd.notna(value):
                    lookup[str(value)] = model_id

    if "model_id" not in training_defaults.columns and "basemodel" in training_defaults.columns:
        training_defaults["model_id"] = training_defaults["basemodel"].map(
            lambda value: lookup.get(str(value), str(value)) if pd.notna(value) else None
        )
    elif "model_id" in training_defaults.columns:
        training_defaults["model_id"] = training_defaults["model_id"].map(
            lambda value: lookup.get(str(value), str(value)) if pd.notna(value) else None
        )

    if "basemodel" not in training_defaults.columns and "model_id" in training_defaults.columns:
        training_defaults["basemodel"] = training_defaults["model_id"]

    numeric_columns = [
        "learning_rate",
        "seq_length_min",
        "seq_length_max",
        "max_token_length",
        "gpu_memory",
        "gradient_accumulation_steps",
        "batch_size",
        "model_complexity",
        "max_token_scaling",
        "tokenization_kmer",
        "tokenization_shift",
    ]
    for col in numeric_columns:
        if col in model_db.columns:
            try:
                model_db[col] = pd.to_numeric(model_db[col])
            except (ValueError, TypeError):
                pass
        if col in training_defaults.columns:
            try:
                training_defaults[col] = pd.to_numeric(training_defaults[col])
            except (ValueError, TypeError):
                pass

    return model_db, training_defaults


def get_tokenize_function(model_name: str) -> Callable:
    model_name_lower = str(model_name).lower()
    if "prokbert" in model_name_lower:
        return tokenize_function_prokbert
    if "nucleotide" in model_name_lower or model_name_lower.startswith("nt"):
        return tokenize_function_NT
    if "dnabert" in model_name_lower:
        return tokenize_function_DNABERT
    if "evo" in model_name_lower or "metagene" in model_name_lower:
        return tokenize_function_evo_metagene
    raise ValueError(f"Unknown model name {model_name}.")


def _extract_labels(examples) -> torch.Tensor:
    if "y" in examples:
        labels = examples["y"]
    elif "labels" in examples:
        labels = examples["labels"]
    else:
        raise KeyError("Expected examples to contain either 'y' or 'labels'.")
    return torch.tensor(labels, dtype=torch.int64)


def _apply_attention_mask_filter(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mask_token_ids: tuple[int, ...] = (2, 3),
) -> torch.Tensor:
    if not mask_token_ids:
        return attention_mask

    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for token_id in mask_token_ids:
        mask |= input_ids == token_id

    attention_mask = attention_mask.clone().detach()
    attention_mask[mask] = 0
    return attention_mask


def _tokenize_common(
    examples,
    tokenizer,
    *,
    max_seq_len: Optional[int] = None,
    padding: Union[bool, str] = "longest",
    mask_token_ids: tuple[int, ...] = (2, 3),
):
    kwargs: Dict[str, Any] = {
        "padding": padding,
        "add_special_tokens": True,
        "return_tensors": "pt",
    }
    if max_seq_len is not None:
        kwargs["truncation"] = True
        kwargs["max_length"] = int(max_seq_len)

    encoded = tokenizer(examples["segment"], **kwargs)
    input_ids = encoded["input_ids"].clone().detach()
    attention_mask = encoded["attention_mask"].clone().detach()
    attention_mask = _apply_attention_mask_filter(
        input_ids,
        attention_mask,
        mask_token_ids=mask_token_ids,
    )
    labels = _extract_labels(examples)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def tokenize_function_prokbert(examples, tokenizer, max_seq_len: Optional[int] = None):
    return _tokenize_common(
        examples,
        tokenizer,
        max_seq_len=max_seq_len,
        padding=True if max_seq_len is None else "longest",
        mask_token_ids=(2, 3),
    )


def tokenize_function_NT(examples, tokenizer, max_seq_len: Optional[int] = None):
    return _tokenize_common(
        examples,
        tokenizer,
        max_seq_len=max_seq_len,
        padding="longest",
        mask_token_ids=(2, 3),
    )


def tokenize_function_DNABERT(examples, tokenizer, max_seq_len: Optional[int] = None):
    return _tokenize_common(
        examples,
        tokenizer,
        max_seq_len=max_seq_len,
        padding="longest",
        mask_token_ids=(2, 3),
    )


def tokenize_function_evo_metagene(examples, tokenizer, max_seq_len: Optional[int] = None):
    return _tokenize_common(
        examples,
        tokenizer,
        max_seq_len=max_seq_len,
        padding="longest",
        mask_token_ids=(),
    )


class TrainingHelper:
    """
    HF-dataset-backed training helper.

    This refactored helper reads only from a Hugging Face dataset repository.
    It does not support loading a local spreadsheet or local CSV/JSON registry path.

    Expected files in the dataset repository:
    - basemodels.json
    - default_training_parameters.json

    The JSON files can either be:
    - a list of row objects
    - or an object with a top-level ``records`` list
    """

    training_parameters = [
        "learning_rate",
        "batch_size",
        "gradient_accumulation_steps",
        "max_token_length",
    ]

    parameter_group_mappings = {
        "ls": "Ls",
        "ep": "epochs",
        "lr": "learning_rate",
        "bs": "batch_size",
        "gac": "gradient_accumulation_steps",
        "mtl": "max_token_length",
    }

    group_mappings_to_params = {v: k for k, v in parameter_group_mappings.items()}
    parameter_group_sep = "___"

    def __init__(
        self,
        *,
        repo_id: str = DEFAULT_REGISTRY_REPO_ID,
        revision: str = DEFAULT_REGISTRY_REVISION,
        token: Optional[str] = None,
        local_files_only: bool = False,
    ) -> None:
        self.repo_id = repo_id
        self.revision = revision
        self.token = token
        self.local_files_only = local_files_only

        self.model_db, self.finetuning_default_params = _load_registry_from_hf_dataset(
            repo_id=repo_id,
            revision=revision,
            token=token,
            local_files_only=local_files_only,
        )
        self._build_indexes()

    def _build_indexes(self) -> None:
        self.basemodels = set(self.model_db["model_id"].astype(str))
        self._model_lookup_columns = [
            col for col in ("model_id", "name", "hf_name", "hf_path", "hf_repo_id")
            if col in self.model_db.columns
        ]

        self._model_ref_to_id: dict[str, str] = {}
        for _, row in self.model_db.iterrows():
            model_id = str(row["model_id"])
            for key in self._model_lookup_columns:
                value = row.get(key)
                if pd.notna(value):
                    self._model_ref_to_id[str(value)] = model_id

    def list_models(self) -> pd.DataFrame:
        columns = [
            col for col in (
                "model_id",
                "name",
                "hf_name",
                "hf_path",
                "model_complexity",
                "tokenizer_short_name",
                "train_tokenizer_function",
            )
            if col in self.model_db.columns
        ]
        return self.model_db[columns].copy()

    def _resolve_model_id(self, model_ref: str) -> str:
        model_ref = str(model_ref)
        if model_ref in self._model_ref_to_id:
            return self._model_ref_to_id[model_ref]

        raise ValueError(
            f"Unknown model reference '{model_ref}'. "
            f"Supported model ids are: {sorted(self.basemodels)}"
        )

    def _resolve_model_row(self, model_ref: str) -> pd.Series:
        model_id = self._resolve_model_id(model_ref)
        rows = self.model_db[self.model_db["model_id"].astype(str) == model_id]
        if rows.empty:
            raise ValueError(f"Model '{model_ref}' resolved to '{model_id}' but no metadata row was found.")
        return rows.iloc[0]

    def get_my_finetunig_model_name(
        self,
        prefix: str,
        short_name: str,
        dataset: str,
        learning_rate=None,
        epochs=None,
        gradient_accumulation_steps=None,
        Ls=None,
        batch_size=None,
        max_token_length=None,
    ) -> str:
        parts = [prefix, short_name, dataset]
        params = {
            "Ls": Ls,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "max_token_length": max_token_length,
        }

        for param_name, value in params.items():
            if value is not None:
                abbr = self.group_mappings_to_params.get(param_name, param_name)
                parts.append(f"{abbr}{value}")

        return self.parameter_group_sep.join(parts)

    def get_my_finetuning_model_name(self, *args, **kwargs) -> str:
        return self.get_my_finetunig_model_name(*args, **kwargs)

    def _match_training_defaults(
        self,
        model_ref: str,
        *,
        seq_len_bp: int,
        task_type: str = "sequence_classification",
        system: Optional[str] = None,
        gpu_memory: Optional[int] = None,
        dtype: Optional[str] = None,
    ) -> pd.DataFrame:
        model_id = self._resolve_model_id(model_ref)

        candidates = self.finetuning_default_params[
            (self.finetuning_default_params["model_id"].astype(str) == str(model_id))
            & (self.finetuning_default_params["seq_length_min"] <= int(seq_len_bp))
            & (self.finetuning_default_params["seq_length_max"] >= int(seq_len_bp))
        ].copy()

        if "task_type" in candidates.columns and task_type is not None:
            task_matches = candidates[candidates["task_type"].astype(str) == str(task_type)]
            if not task_matches.empty:
                candidates = task_matches

        if system is not None and "system" in candidates.columns:
            system_matches = candidates[candidates["system"].astype(str) == str(system)]
            if not system_matches.empty:
                candidates = system_matches

        if gpu_memory is not None and "gpu_memory" in candidates.columns:
            memory_matches = candidates[candidates["gpu_memory"] == gpu_memory]
            if not memory_matches.empty:
                candidates = memory_matches

        if dtype is not None and "dtype" in candidates.columns:
            dtype_matches = candidates[candidates["dtype"].astype(str) == str(dtype)]
            if not dtype_matches.empty:
                candidates = dtype_matches

        return candidates

    def get_my_training_parameters(
        self,
        model: str,
        actLs: int = 512,
        task_type: str = "sequence_classification",
        system: Optional[str] = None,
        gpu_memory: Optional[int] = None,
        dtype: Optional[str] = None,
    ) -> Dict[str, Any]:
        data_answer = self._match_training_defaults(
            model,
            seq_len_bp=actLs,
            task_type=task_type,
            system=system,
            gpu_memory=gpu_memory,
            dtype=dtype,
        )

        if data_answer.empty:
            raise ValueError(
                f"No training parameters found for model '{model}' with actLs={actLs}, "
                f"task_type='{task_type}', system={system}, gpu_memory={gpu_memory}, dtype={dtype}."
            )

        row = data_answer.iloc[0]
        params_dict = row[self.training_parameters].to_dict()

        for key in ("batch_size", "gradient_accumulation_steps", "max_token_length"):
            if key in params_dict and pd.notna(params_dict[key]):
                params_dict[key] = int(params_dict[key])

        if "learning_rate" in params_dict and pd.notna(params_dict["learning_rate"]):
            params_dict["learning_rate"] = float(params_dict["learning_rate"])

        for key in ("dtype", "system", "task_type", "model_id", "basemodel"):
            if key in row and pd.notna(row[key]):
                params_dict[key] = str(row[key])

        if "gpu_memory" in row and pd.notna(row["gpu_memory"]):
            params_dict["gpu_memory"] = int(row["gpu_memory"])

        return params_dict

    def parse_model_name(self, model_name: str) -> Dict[str, Union[str, int, float]]:
        parts = model_name.split(self.parameter_group_sep)
        if len(parts) < 3:
            raise ValueError("Model name must contain at least prefix, short name, and dataset.")

        result: Dict[str, Union[str, int, float]] = {
            "prefix": parts[0],
            "base_model": parts[1],
            "task": parts[2],
        }

        param_keys = list(self.parameter_group_mappings.keys())
        pattern_keys = "|".join(re.escape(key) for key in param_keys)
        pattern = re.compile(
            r"^(?P<abbr>(" + pattern_keys + r"))_?(?P<value>[-+]?\d*\.?\d+(?:e[-+]?\d+)?)$",
            re.IGNORECASE,
        )

        for part in parts[3:]:
            match = pattern.match(part)
            if not match:
                continue

            abbr = match.group("abbr").lower()
            value_str = match.group("value")
            if "." in value_str or "e" in value_str.lower():
                value: Union[int, float] = float(value_str)
            else:
                value = int(value_str)

            full_param = self.parameter_group_mappings.get(abbr, abbr)
            result[full_param] = value

        return result

    def register_all_models(self, models_path: str) -> pd.DataFrame:
        if not os.path.exists(models_path) or not os.listdir(models_path):
            raise ValueError(f"The provided models_path '{models_path}' does not exist or is empty.")

        records: list[dict[str, Any]] = []
        model_dirs = [d for d in os.listdir(models_path) if os.path.isdir(os.path.join(models_path, d))]

        for model_dir in model_dirs:
            model_dir_path = os.path.join(models_path, model_dir)
            try:
                metadata = self.parse_model_name(model_dir)
            except Exception:
                continue

            checkpoint_dirs = [
                d
                for d in os.listdir(model_dir_path)
                if os.path.isdir(os.path.join(model_dir_path, d)) and "checkpoint-" in d
            ]

            for checkpoint_dir in checkpoint_dirs:
                cp_match = re.search(r"checkpoint-(\d+)", checkpoint_dir)
                if cp_match is None:
                    continue

                checkpoint_number = int(cp_match.group(1))
                checkpoint_path = os.path.join(model_dir_path, checkpoint_dir)

                record = metadata.copy()
                record["checkpoint"] = checkpoint_number
                record["checkpoint_path"] = checkpoint_path
                record["model_directory"] = model_dir
                records.append(record)

        return pd.DataFrame(records)

    def select_preferred_checkpoints(self, df: pd.DataFrame) -> pd.DataFrame:
        preferred_records = []
        for model_dir, group in df.groupby("model_directory"):
            if (group["checkpoint"] == 0).any():
                selected = group[group["checkpoint"] == 0].iloc[0]
            else:
                selected = group.loc[group["checkpoint"].idxmax()]
            preferred_records.append(selected)

        return pd.DataFrame(preferred_records)

    def get_tokenizer_for_basemodel(
        self,
        basemodel: str,
        *,
        trust_remote_code: bool = True,
        local_files_only: bool = False,
    ):
        model_row = self._resolve_model_row(basemodel)

        if "hf_path" not in model_row or pd.isna(model_row["hf_path"]):
            raise ValueError(f"No hf_path found for model '{basemodel}'.")

        hf_path = str(model_row["hf_path"])
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            hf_path,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )

    def get_tokenizer_short_name(self, model_ref: str) -> str:
        model_row = self._resolve_model_row(model_ref)
        tokenizer_short_name = model_row.get("tokenizer_short_name", None)
        if tokenizer_short_name is None or (isinstance(tokenizer_short_name, float) and pd.isna(tokenizer_short_name)):
            raise ValueError(f"No tokenizer_short_name found for model '{model_ref}'.")
        return str(tokenizer_short_name)

    def get_max_token_scaling(self, base_name: str) -> float:
        model_row = self._resolve_model_row(base_name)
        value = model_row.get("max_token_scaling", None)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            raise ValueError(f"No max_token_scaling found for model '{base_name}'.")
        return float(value)

    def get_tokenize_function(self, model_ref: str) -> Callable:
        model_row = self._resolve_model_row(model_ref)
        fn_name = str(model_row.get("train_tokenizer_function", "")).strip().lower()

        mapping = {
            "tokenize_function_prokbert": tokenize_function_prokbert,
            "tokenize_function_nt": tokenize_function_NT,
            "tokenize_function_dnabert": tokenize_function_DNABERT,
            "tokenize_function_evo_metagene": tokenize_function_evo_metagene,
        }

        if fn_name in mapping:
            return mapping[fn_name]

        return get_tokenize_function(str(model_ref))

    def build_tokenize_callable(
        self,
        model_ref: str,
        tokenizer,
        *,
        max_seq_len: Optional[int] = None,
    ) -> Callable:
        tokenize_fn = self.get_tokenize_function(model_ref)

        def _callable(examples):
            return tokenize_fn(examples, tokenizer, max_seq_len=max_seq_len)

        return _callable
