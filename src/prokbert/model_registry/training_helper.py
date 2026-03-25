from __future__ import annotations

import gc
import json
import math
import os
import re
import secrets
import string
from copy import deepcopy
from dataclasses import MISSING, fields, is_dataclass
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



def _slugify_name_component(value: Any) -> str:
    text = str(value).strip()
    text = text.replace("/", "-").replace(" ", "-")
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text)
    text = text.strip("-_")
    return text or "na"


def _format_name_value(value: Any) -> str:
    if isinstance(value, bool):
        text = "1" if value else "0"
    elif isinstance(value, int):
        text = str(value)
    elif isinstance(value, float):
        if math.isfinite(value):
            text = format(value, ".4g")
        else:
            text = "nan"
    else:
        text = str(value)
    return _slugify_name_component(text)


def _normalize_scalar_for_compare(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if hasattr(value, "value") and not isinstance(value, (str, bytes, int, float, bool)):
        try:
            return _normalize_scalar_for_compare(value.value)
        except Exception:
            pass
    if hasattr(value, "item") and callable(getattr(value, "item", None)):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _values_equal(left: Any, right: Any) -> bool:
    left = _normalize_scalar_for_compare(left)
    right = _normalize_scalar_for_compare(right)

    if left is None and right is None:
        return True
    if isinstance(left, float) and isinstance(right, float):
        if math.isnan(left) and math.isnan(right):
            return True
    return left == right


def _json_safe(value: Any) -> Any:
    value = _normalize_scalar_for_compare(value)

    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return float(value)
    if isinstance(value, str):
        return value
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    if hasattr(value, "tolist") and callable(getattr(value, "tolist", None)):
        try:
            return _json_safe(value.tolist())
        except Exception:
            pass
    return str(value)


def _series_to_clean_dict(series: pd.Series) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in series.items():
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        result[str(key)] = _json_safe(value)
    return result


def _training_args_to_dict(training_args: Any) -> Dict[str, Any]:
    if training_args is None:
        return {}
    if isinstance(training_args, dict):
        return deepcopy(training_args)
    if is_dataclass(training_args):
        return {field.name: getattr(training_args, field.name) for field in fields(training_args)}
    if hasattr(training_args, "to_dict") and callable(getattr(training_args, "to_dict", None)):
        return dict(training_args.to_dict())
    if hasattr(training_args, "__dict__"):
        return {
            key: value
            for key, value in vars(training_args).items()
            if not key.startswith("_")
        }
    raise TypeError(
        "training_args must be a dict-like object, dataclass instance, or expose to_dict()/__dict__."
    )


def _extract_non_default_training_args(training_args: Any) -> Dict[str, Any]:
    if training_args is None:
        return {}

    if isinstance(training_args, dict):
        return {
            str(key): value
            for key, value in training_args.items()
            if value is not None
        }

    if is_dataclass(training_args):
        result: Dict[str, Any] = {}
        for field in fields(training_args):
            if field.name.startswith("_"):
                continue
            value = getattr(training_args, field.name)

            if field.default is not MISSING:
                default_value = field.default
            elif field.default_factory is not MISSING:  # type: ignore[attr-defined]
                default_value = field.default_factory()  # type: ignore[misc]
            else:
                default_value = MISSING

            if default_value is MISSING:
                if value is not None:
                    result[field.name] = value
            else:
                if not _values_equal(value, default_value):
                    result[field.name] = value
        return result

    return {
        str(key): value
        for key, value in _training_args_to_dict(training_args).items()
        if value is not None
    }


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
    run_id_prefix = "uid"
    training_profile_filename = "training_profile.json"
    training_argument_aliases = {
        "learning_rate": "learning_rate",
        "per_device_train_batch_size": "batch_size",
        "gradient_accumulation_steps": "gradient_accumulation_steps",
        "num_train_epochs": "epochs",
        "lr_scheduler_type": "learning_type",
        "weight_decay": "weight_decay",
        "warmup_ratio": "warmup_ratio",
        "warmup_steps": "warmup_steps",
        "max_steps": "max_steps",
        "save_steps": "save_steps",
        "eval_steps": "eval_steps",
        "logging_steps": "logging_steps",
        "seed": "seed",
        "gradient_checkpointing": "gradient_checkpointing",
        "optim": "optim",
        "per_device_eval_batch_size": "eval_batch_size",
    }

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

    @staticmethod
    def _generate_run_id(length: int = 8) -> str:
        if int(length) <= 0:
            raise ValueError("length must be a positive integer.")
        alphabet = string.ascii_lowercase + string.digits
        return "".join(secrets.choice(alphabet) for _ in range(int(length)))

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
        unique_id: Optional[str] = None,
        include_unique_id: bool = True,
        unique_id_length: int = 8,
    ) -> str:
        parts = [
            _slugify_name_component(prefix),
            _slugify_name_component(short_name),
            _slugify_name_component(dataset),
        ]
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
                parts.append(f"{abbr}{_format_name_value(value)}")

        if include_unique_id:
            resolved_unique_id = _slugify_name_component(
                unique_id if unique_id is not None else self._generate_run_id(unique_id_length)
            )
            parts.append(f"{self.run_id_prefix}{resolved_unique_id}")

        return self.parameter_group_sep.join(parts)

    def get_my_finetuning_model_name(self, *args, **kwargs) -> str:
        return self.get_my_finetunig_model_name(*args, **kwargs)

    def build_training_run_name(
        self,
        model: str,
        *,
        dataset: str,
        prefix: str = "FT",
        actLs: int = 512,
        task_type: str = "sequence_classification",
        system: Optional[str] = None,
        gpu_memory: Optional[int] = None,
        dtype: Optional[str] = None,
        learning_rate: Optional[float] = None,
        epochs: Optional[Union[int, float]] = None,
        gradient_accumulation_steps: Optional[int] = None,
        batch_size: Optional[int] = None,
        max_token_length: Optional[int] = None,
        unique_id: Optional[str] = None,
        include_unique_id: bool = True,
        unique_id_length: int = 8,
    ) -> str:
        defaults = self.get_my_training_parameters(
            model=model,
            actLs=actLs,
            task_type=task_type,
            system=system,
            gpu_memory=gpu_memory,
            dtype=dtype,
        )
        model_id = self._resolve_model_id(model)

        return self.get_my_finetunig_model_name(
            prefix=prefix,
            short_name=model_id,
            dataset=dataset,
            learning_rate=learning_rate if learning_rate is not None else defaults.get("learning_rate"),
            epochs=epochs,
            gradient_accumulation_steps=(
                gradient_accumulation_steps
                if gradient_accumulation_steps is not None
                else defaults.get("gradient_accumulation_steps")
            ),
            Ls=actLs,
            batch_size=batch_size if batch_size is not None else defaults.get("batch_size"),
            max_token_length=(
                max_token_length if max_token_length is not None else defaults.get("max_token_length")
            ),
            unique_id=unique_id,
            include_unique_id=include_unique_id,
            unique_id_length=unique_id_length,
        )

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

        run_id_pattern = re.compile(
            rf"^(?P<abbr>{re.escape(self.run_id_prefix)})_?(?P<value>[A-Za-z0-9._-]+)$",
            re.IGNORECASE,
        )
        param_keys = list(self.parameter_group_mappings.keys())
        pattern_keys = "|".join(re.escape(key) for key in param_keys)
        numeric_pattern = re.compile(
            r"^(?P<abbr>(" + pattern_keys + r"))_?(?P<value>[-+]?\d*\.?\d+(?:e[-+]?\d+)?)$",
            re.IGNORECASE,
        )

        extras: list[str] = []
        for part in parts[3:]:
            run_id_match = run_id_pattern.match(part)
            if run_id_match:
                result["run_id"] = run_id_match.group("value")
                continue

            numeric_match = numeric_pattern.match(part)
            if numeric_match:
                abbr = numeric_match.group("abbr").lower()
                value_str = numeric_match.group("value")
                if "." in value_str or "e" in value_str.lower():
                    value: Union[int, float] = float(value_str)
                else:
                    value = int(value_str)

                full_param = self.parameter_group_mappings.get(abbr, abbr)
                result[full_param] = value
                continue

            extras.append(part)

        if extras:
            result["extras"] = extras

        return result

    def _canonicalize_training_arg_overrides(
        self,
        training_args: Optional[Any],
        *,
        defaults: Optional[Dict[str, Any]] = None,
        include_raw: bool = False,
    ) -> Dict[str, Any]:
        raw_args = _extract_non_default_training_args(training_args)
        if not raw_args:
            return {"canonical": {}, "raw": {} if include_raw else {}}

        defaults = defaults or {}
        canonical: Dict[str, Any] = {}

        def _maybe_add(target_key: str, value: Any) -> None:
            if target_key in defaults and _values_equal(value, defaults[target_key]):
                return
            canonical[target_key] = _json_safe(value)

        for raw_key, raw_value in raw_args.items():
            if raw_key in self.training_argument_aliases:
                _maybe_add(self.training_argument_aliases[raw_key], raw_value)

        bf16_value = raw_args.get("bf16", None)
        fp16_value = raw_args.get("fp16", None)
        if bf16_value is True:
            _maybe_add("dtype", "bf16")
        elif fp16_value is True:
            _maybe_add("dtype", "fp16")

        result: Dict[str, Any] = {"canonical": canonical}
        if include_raw:
            result["raw"] = {str(key): _json_safe(value) for key, value in raw_args.items()}
        return result

    def get_training_profile(
        self,
        model: str,
        actLs: int = 512,
        *,
        dataset: Optional[str] = None,
        prefix: str = "FT",
        task_type: str = "sequence_classification",
        system: Optional[str] = None,
        gpu_memory: Optional[int] = None,
        dtype: Optional[str] = None,
        learning_rate: Optional[float] = None,
        epochs: Optional[Union[int, float]] = None,
        gradient_accumulation_steps: Optional[int] = None,
        batch_size: Optional[int] = None,
        max_token_length: Optional[int] = None,
        overrides: Optional[Dict[str, Any]] = None,
        training_args: Optional[Any] = None,
        run_name: Optional[str] = None,
        unique_run_name: bool = True,
        unique_id: Optional[str] = None,
        unique_id_length: int = 8,
        include_raw_training_args: bool = False,
    ) -> Dict[str, Any]:
        model_row = self._resolve_model_row(model)
        model_id = str(model_row["model_id"])
        matched_rows = self._match_training_defaults(
            model,
            seq_len_bp=actLs,
            task_type=task_type,
            system=system,
            gpu_memory=gpu_memory,
            dtype=dtype,
        )
        if matched_rows.empty:
            raise ValueError(
                f"No training parameters found for model '{model}' with actLs={actLs}, "
                f"task_type='{task_type}', system={system}, gpu_memory={gpu_memory}, dtype={dtype}."
            )
        selected_row = matched_rows.iloc[0]
        defaults = self.get_my_training_parameters(
            model=model,
            actLs=actLs,
            task_type=task_type,
            system=system,
            gpu_memory=gpu_memory,
            dtype=dtype,
        )

        manual_overrides: Dict[str, Any] = {}
        direct_values = {
            "learning_rate": learning_rate,
            "epochs": epochs,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "batch_size": batch_size,
            "max_token_length": max_token_length,
        }
        for key, value in direct_values.items():
            if value is None:
                continue
            if key in defaults and _values_equal(value, defaults[key]):
                continue
            manual_overrides[key] = _json_safe(value)

        if overrides:
            for key, value in overrides.items():
                if value is None:
                    continue
                if key in defaults and _values_equal(value, defaults[key]):
                    continue
                manual_overrides[str(key)] = _json_safe(value)

        training_arg_info = self._canonicalize_training_arg_overrides(
            training_args,
            defaults=defaults,
            include_raw=include_raw_training_args,
        )
        training_arg_overrides = training_arg_info.get("canonical", {})
        manual_overrides = {
            key: value
            for key, value in manual_overrides.items()
            if key not in training_arg_overrides or not _values_equal(value, training_arg_overrides[key])
        }

        effective = {str(key): _json_safe(value) for key, value in defaults.items()}
        effective.update(training_arg_overrides)
        effective.update(manual_overrides)

        resolved_dataset = dataset if dataset is not None else task_type
        if run_name is None:
            run_name = self.get_my_finetunig_model_name(
                prefix=prefix,
                short_name=model_id,
                dataset=resolved_dataset,
                learning_rate=effective.get("learning_rate"),
                epochs=effective.get("epochs"),
                gradient_accumulation_steps=effective.get("gradient_accumulation_steps"),
                Ls=actLs,
                batch_size=effective.get("batch_size"),
                max_token_length=effective.get("max_token_length"),
                unique_id=unique_id,
                include_unique_id=unique_run_name,
                unique_id_length=unique_id_length,
            )

        parsed_run = self.parse_model_name(run_name)
        profile: Dict[str, Any] = {
            "format_name": "prokbert_training_profile",
            "format_version": 1,
            "registry": {
                "repo_id": self.repo_id,
                "revision": self.revision,
            },
            "run": {
                "name": run_name,
                "run_id": parsed_run.get("run_id"),
                "prefix": _json_safe(prefix),
                "dataset": _json_safe(resolved_dataset),
            },
            "selection": {
                "model_ref": _json_safe(model),
                "model_id": model_id,
                "seq_len_bp": int(actLs),
                "task_type": _json_safe(task_type),
                "system": _json_safe(system if system is not None else defaults.get("system")),
                "gpu_memory": _json_safe(gpu_memory if gpu_memory is not None else defaults.get("gpu_memory")),
                "dtype": _json_safe(dtype if dtype is not None else defaults.get("dtype")),
            },
            "model": {
                "main": {
                    key: _json_safe(value)
                    for key, value in {
                        "model_id": model_row.get("model_id"),
                        "name": model_row.get("name"),
                        "hf_name": model_row.get("hf_name"),
                        "hf_path": model_row.get("hf_path"),
                        "tokenizer_short_name": model_row.get("tokenizer_short_name"),
                        "tokenization": model_row.get("tokenization"),
                        "tokenization_kmer": model_row.get("tokenization_kmer"),
                        "tokenization_shift": model_row.get("tokenization_shift"),
                        "model_complexity": model_row.get("model_complexity"),
                        "max_token_scaling": model_row.get("max_token_scaling"),
                        "train_tokenizer_function": model_row.get("train_tokenizer_function"),
                    }.items()
                    if value is not None and not (isinstance(value, float) and math.isnan(value))
                },
                "registry_row": _series_to_clean_dict(model_row),
            },
            "defaults": {
                "main": {str(key): _json_safe(value) for key, value in defaults.items()},
                "registry_row": _series_to_clean_dict(selected_row),
            },
            "overrides": {
                "training_args": training_arg_overrides,
                "explicit": manual_overrides,
            },
            "effective": effective,
        }

        if include_raw_training_args and training_arg_info.get("raw"):
            profile["overrides"]["raw_training_args"] = training_arg_info["raw"]

        return profile

    def save_training_profile(
        self,
        save_path: Union[str, Path],
        training_profile: Dict[str, Any],
        *,
        filename: str = training_profile_filename,
        indent: int = 2,
    ) -> str:
        path = Path(save_path)
        if path.suffix.lower() == ".json":
            file_path = path
            file_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)
            file_path = path / filename

        with open(file_path, "w", encoding="utf-8") as handle:
            json.dump(_json_safe(training_profile), handle, indent=int(indent), sort_keys=True, ensure_ascii=False)
            handle.write("\n")

        return str(file_path)

    @classmethod
    def load_training_profile(
        cls,
        load_path: Union[str, Path],
        *,
        filename: str = training_profile_filename,
    ) -> Dict[str, Any]:
        path = Path(load_path)
        if path.is_dir():
            path = path / filename
        if not path.exists():
            raise FileNotFoundError(f"Training profile JSON not found: {path}")

        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if not isinstance(payload, dict):
            raise ValueError("Training profile JSON must contain a top-level object.")
        return payload

    read_training_profile = load_training_profile
    save_pretrained_training_profile = save_training_profile

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


    @staticmethod
    def _normalize_cuda_device(device: Optional[Union[str, int, torch.device]] = None) -> torch.device:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available on this system.")

        if device is None:
            return torch.device(f"cuda:{torch.cuda.current_device()}")

        if isinstance(device, torch.device):
            if device.type != "cuda":
                raise ValueError(f"Expected a CUDA device, got {device}.")
            if device.index is None:
                return torch.device(f"cuda:{torch.cuda.current_device()}")
            return device

        if isinstance(device, int):
            return torch.device(f"cuda:{device}")

        device_str = str(device)
        if not device_str.startswith("cuda"):
            raise ValueError(f"Expected a CUDA device string like 'cuda:0', got {device_str!r}.")
        return torch.device(device_str)

    @staticmethod
    def _coerce_torch_dtype(dtype: Optional[Union[str, torch.dtype]]) -> Optional[torch.dtype]:
        if dtype is None or isinstance(dtype, torch.dtype):
            return dtype

        mapping = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "half": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
            "float": torch.float32,
            "fp64": torch.float64,
            "float64": torch.float64,
            "double": torch.float64,
        }
        key = str(dtype).strip().lower()
        if key not in mapping:
            return None
        return mapping[key]

    @staticmethod
    def _dtype_num_bytes(dtype: Optional[Union[str, torch.dtype]]) -> int:
        dtype_obj = TrainingHelper._coerce_torch_dtype(dtype)
        if dtype_obj in (torch.float16, torch.bfloat16):
            return 2
        if dtype_obj in (torch.float32, torch.int32):
            return 4
        if dtype_obj in (torch.float64, torch.int64):
            return 8
        return 4

    @staticmethod
    def _call_model(model, batch):
        if isinstance(batch, dict):
            return model(**batch)
        if isinstance(batch, (tuple, list)):
            return model(*batch)
        return model(batch)

    @staticmethod
    def _extract_loss(outputs, loss_extractor: Optional[Callable] = None):
        if loss_extractor is not None:
            return loss_extractor(outputs)

        if hasattr(outputs, "loss") and outputs.loss is not None:
            return outputs.loss

        if isinstance(outputs, dict) and "loss" in outputs:
            return outputs["loss"]

        if isinstance(outputs, (tuple, list)) and outputs:
            first = outputs[0]
            if torch.is_tensor(first) and first.ndim == 0:
                return first

        raise ValueError(
            "Could not infer a scalar loss from model outputs. "
            "Pass loss_extractor=... to autotune_batch_size()."
        )

    @staticmethod
    def _round_down_to_multiple(value: int, multiple_of: int) -> int:
        if multiple_of <= 1:
            return int(value)
        return int(value) - (int(value) % int(multiple_of))

    def get_gpu_memory_info(
        self,
        device: Optional[Union[str, int, torch.device]] = None,
    ) -> Dict[str, Any]:
        """
        Return basic runtime CUDA memory information for the selected GPU.
        """
        device_obj = self._normalize_cuda_device(device)
        torch.cuda.synchronize(device_obj)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_obj)

        return {
            "device": str(device_obj),
            "device_name": torch.cuda.get_device_name(device_obj),
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
            "free_gb": float(free_bytes) / (1024 ** 3),
            "total_gb": float(total_bytes) / (1024 ** 3),
            "allocated_bytes": int(torch.cuda.memory_allocated(device_obj)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device_obj)),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device_obj)),
        }

    def estimate_model_parameter_bytes(
        self,
        *,
        model=None,
        model_ref: Optional[str] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        only_trainable: bool = False,
        include_buffers: bool = True,
    ) -> int:
        """
        Estimate the in-memory size of model parameters and optional buffers.

        Preferred mode:
        - pass a loaded PyTorch/HF model instance in ``model``.

        Fallback mode:
        - pass ``model_ref`` so the helper can use ``model_complexity`` from the
          HF registry as an approximate parameter count in millions.
        """
        if model is not None:
            if hasattr(model, "get_memory_footprint"):
                try:
                    return int(model.get_memory_footprint(return_buffers=include_buffers))
                except TypeError:
                    try:
                        return int(model.get_memory_footprint())
                    except Exception:
                        pass
                except Exception:
                    pass

            total_bytes = 0
            for parameter in model.parameters():
                if only_trainable and not parameter.requires_grad:
                    continue
                total_bytes += parameter.nelement() * parameter.element_size()

            if include_buffers:
                for buffer in model.buffers():
                    total_bytes += buffer.nelement() * buffer.element_size()

            return int(total_bytes)

        if model_ref is None:
            raise ValueError("Provide either model=... or model_ref=... .")

        model_row = self._resolve_model_row(model_ref)
        model_complexity = model_row.get("model_complexity", None)
        if model_complexity is None or pd.isna(model_complexity):
            raise ValueError(
                f"No model_complexity found in the registry for model '{model_ref}'."
            )

        bytes_per_parameter = self._dtype_num_bytes(dtype)
        estimated_parameter_count = float(model_complexity) * 1_000_000.0
        return int(estimated_parameter_count * bytes_per_parameter)

    def _select_reference_defaults_row(
        self,
        model_ref: str,
        *,
        seq_len_bp: int,
        task_type: str = "sequence_classification",
        system: Optional[str] = None,
        gpu_memory: Optional[int] = None,
        dtype: Optional[str] = None,
    ) -> pd.Series:
        model_id = self._resolve_model_id(model_ref)
        candidates = self.finetuning_default_params[
            self.finetuning_default_params["model_id"].astype(str) == str(model_id)
        ].copy()

        if candidates.empty:
            raise ValueError(f"No training defaults found for model '{model_ref}'.")

        if "task_type" in candidates.columns and task_type is not None:
            matches = candidates[candidates["task_type"].astype(str) == str(task_type)]
            if not matches.empty:
                candidates = matches

        if system is not None and "system" in candidates.columns:
            matches = candidates[candidates["system"].astype(str) == str(system)]
            if not matches.empty:
                candidates = matches

        if dtype is not None and "dtype" in candidates.columns:
            matches = candidates[candidates["dtype"].astype(str) == str(dtype)]
            if not matches.empty:
                candidates = matches

        if "seq_length_min" in candidates.columns and "seq_length_max" in candidates.columns:
            def _seq_distance(row: pd.Series) -> int:
                lower = int(row["seq_length_min"])
                upper = int(row["seq_length_max"])
                if lower <= int(seq_len_bp) <= upper:
                    return 0
                if int(seq_len_bp) < lower:
                    return lower - int(seq_len_bp)
                return int(seq_len_bp) - upper

            candidates["_seq_distance"] = candidates.apply(_seq_distance, axis=1)
        else:
            candidates["_seq_distance"] = 0

        if gpu_memory is not None and "gpu_memory" in candidates.columns:
            numeric_gpu_memory = pd.to_numeric(candidates["gpu_memory"], errors="coerce")
            candidates["_gpu_distance"] = (numeric_gpu_memory - float(gpu_memory)).abs()
        else:
            candidates["_gpu_distance"] = 0.0

        sort_columns = ["_seq_distance", "_gpu_distance"]
        candidates = candidates.sort_values(sort_columns, ascending=[True, True])
        return candidates.iloc[0]

    def suggest_batch_size(
        self,
        model: str,
        actLs: int,
        *,
        device: Optional[Union[str, int, torch.device]] = None,
        task_type: str = "sequence_classification",
        system: Optional[str] = None,
        dtype: Optional[str] = None,
        safety_factor: float = 0.85,
        min_batch_size: int = 1,
        max_batch_size: Optional[int] = None,
        multiple_of: int = 1,
        target_effective_batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Suggest a per-device batch size by scaling the registry default to the
        currently available free GPU memory.

        This is a fast heuristic. For high-confidence tuning, follow it with
        ``autotune_batch_size(...)``.
        """
        if not (0.0 < safety_factor <= 1.0):
            raise ValueError("safety_factor must be in the interval (0, 1].")

        memory_info = self.get_gpu_memory_info(device=device)
        reference_row = self._select_reference_defaults_row(
            model,
            seq_len_bp=actLs,
            task_type=task_type,
            system=system,
            dtype=dtype,
        )

        target_params = self.get_my_training_parameters(
            model=model,
            actLs=actLs,
            task_type=task_type,
            dtype=dtype,
        )

        reference_batch_size = int(reference_row["batch_size"])
        reference_max_token_length = int(reference_row["max_token_length"])
        target_max_token_length = int(target_params["max_token_length"])

        if "gpu_memory" in reference_row and pd.notna(reference_row["gpu_memory"]):
            reference_gpu_memory_gb = float(reference_row["gpu_memory"])
        else:
            reference_gpu_memory_gb = memory_info["total_gb"]

        if reference_gpu_memory_gb <= 0:
            raise ValueError("Reference gpu_memory must be positive.")

        reference_tokens_per_step = reference_batch_size * reference_max_token_length
        scaled_token_budget = (
            reference_tokens_per_step
            * (memory_info["free_gb"] / reference_gpu_memory_gb)
            * float(safety_factor)
        )

        suggested_batch_size = max(
            int(min_batch_size),
            int(scaled_token_budget // max(1, target_max_token_length)),
        )

        if max_batch_size is not None:
            suggested_batch_size = min(int(max_batch_size), suggested_batch_size)

        suggested_batch_size = max(int(min_batch_size), suggested_batch_size)
        suggested_batch_size = self._round_down_to_multiple(suggested_batch_size, multiple_of)
        if suggested_batch_size < int(min_batch_size):
            suggested_batch_size = int(min_batch_size)

        result: Dict[str, Any] = {
            "model": str(model),
            "seq_len_bp": int(actLs),
            "suggested_batch_size": int(suggested_batch_size),
            "target_max_token_length": int(target_max_token_length),
            "free_gpu_memory_gb": float(memory_info["free_gb"]),
            "total_gpu_memory_gb": float(memory_info["total_gb"]),
            "reference_batch_size": int(reference_batch_size),
            "reference_max_token_length": int(reference_max_token_length),
            "reference_gpu_memory_gb": float(reference_gpu_memory_gb),
            "reference_system": str(reference_row["system"]) if "system" in reference_row and pd.notna(reference_row["system"]) else None,
            "reference_dtype": str(reference_row["dtype"]) if "dtype" in reference_row and pd.notna(reference_row["dtype"]) else None,
            "estimated_model_parameter_bytes": int(self.estimate_model_parameter_bytes(model_ref=model, dtype=dtype)),
        }

        if target_effective_batch_size is not None:
            result["target_effective_batch_size"] = int(target_effective_batch_size)
            result["gradient_accumulation_steps"] = int(
                math.ceil(int(target_effective_batch_size) / max(1, suggested_batch_size))
            )

        return result

    def _probe_batch_size(
        self,
        model,
        batch_builder: Callable[[int, torch.device], Any],
        *,
        batch_size: int,
        device: Optional[Union[str, int, torch.device]] = None,
        mode: str = "train",
        loss_extractor: Optional[Callable] = None,
        optimizer=None,
        optimizer_step: bool = False,
        clear_cache: bool = True,
    ) -> Dict[str, Any]:
        device_obj = self._normalize_cuda_device(device)
        batch = None
        outputs = None
        loss = None

        if clear_cache:
            gc.collect()
            torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device_obj)

        try:
            model.to(device_obj)
            if mode == "train":
                model.train()
                if hasattr(model, "zero_grad"):
                    model.zero_grad(set_to_none=True)
            elif mode == "eval":
                model.eval()
            else:
                raise ValueError("mode must be either 'train' or 'eval'.")

            batch = batch_builder(int(batch_size), device_obj)

            if mode == "train":
                outputs = self._call_model(model, batch)
                loss = self._extract_loss(outputs, loss_extractor=loss_extractor)
                loss.backward()
                if optimizer is not None and optimizer_step:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                elif hasattr(model, "zero_grad"):
                    model.zero_grad(set_to_none=True)
            else:
                with torch.no_grad():
                    outputs = self._call_model(model, batch)

            torch.cuda.synchronize(device_obj)
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_obj)

            return {
                "fits": True,
                "batch_size": int(batch_size),
                "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device_obj)),
                "free_bytes_after": int(free_bytes),
                "total_bytes": int(total_bytes),
            }
        except torch.cuda.OutOfMemoryError as exc:
            if optimizer is not None:
                try:
                    optimizer.zero_grad(set_to_none=True)
                except Exception:
                    pass
            if hasattr(model, "zero_grad"):
                try:
                    model.zero_grad(set_to_none=True)
                except Exception:
                    pass
            if clear_cache:
                gc.collect()
                torch.cuda.empty_cache()
            return {
                "fits": False,
                "batch_size": int(batch_size),
                "error": str(exc),
            }
        finally:
            del batch
            del outputs
            del loss
            if clear_cache:
                gc.collect()
                torch.cuda.empty_cache()

    def autotune_batch_size(
        self,
        model,
        batch_builder: Callable[[int, torch.device], Any],
        *,
        device: Optional[Union[str, int, torch.device]] = None,
        model_ref: Optional[str] = None,
        actLs: Optional[int] = None,
        task_type: str = "sequence_classification",
        system: Optional[str] = None,
        dtype: Optional[str] = None,
        initial_batch_size: Optional[int] = None,
        min_batch_size: int = 1,
        max_batch_size: Optional[int] = None,
        multiple_of: int = 1,
        target_effective_batch_size: Optional[int] = None,
        mode: str = "train",
        loss_extractor: Optional[Callable] = None,
        optimizer=None,
        optimizer_step: bool = False,
        safety_factor: float = 0.85,
    ) -> Dict[str, Any]:
        """
        Empirically find the largest fitting batch size on the current GPU.

        Parameters
        ----------
        model:
            A loaded torch/transformers model.
        batch_builder:
            Callable with signature ``batch_builder(batch_size, device) -> batch``.
            The returned object must be directly consumable by the model.
        mode:
            ``"train"`` runs forward + backward and is the correct mode for
            estimating training memory. ``"eval"`` runs inference only.
        optimizer / optimizer_step:
            If you pass an optimizer and set ``optimizer_step=True``, the probe
            includes one optimizer step. This is more realistic for Adam-like
            optimizers, but it mutates model weights.

        Returns
        -------
        dict with the selected batch size and probe details.
        """
        if initial_batch_size is None:
            if model_ref is not None and actLs is not None:
                suggestion = self.suggest_batch_size(
                    model=model_ref,
                    actLs=actLs,
                    device=device,
                    task_type=task_type,
                    system=system,
                    dtype=dtype,
                    safety_factor=safety_factor,
                    min_batch_size=min_batch_size,
                    max_batch_size=max_batch_size,
                    multiple_of=multiple_of,
                    target_effective_batch_size=target_effective_batch_size,
                )
                initial_batch_size = int(suggestion["suggested_batch_size"])
            else:
                initial_batch_size = int(min_batch_size)

        initial_batch_size = max(int(min_batch_size), int(initial_batch_size))
        if max_batch_size is not None:
            initial_batch_size = min(initial_batch_size, int(max_batch_size))

        best_probe: Optional[Dict[str, Any]] = None
        probe_history: list[dict[str, Any]] = []

        def _probe(bs: int) -> Dict[str, Any]:
            result = self._probe_batch_size(
                model,
                batch_builder,
                batch_size=int(bs),
                device=device,
                mode=mode,
                loss_extractor=loss_extractor,
                optimizer=optimizer,
                optimizer_step=optimizer_step,
                clear_cache=True,
            )
            probe_history.append(result)
            return result

        first_result = _probe(initial_batch_size)

        if first_result["fits"]:
            best_probe = first_result
            low_good = int(initial_batch_size)
            high_bad: Optional[int] = None
            candidate = int(initial_batch_size)

            while max_batch_size is None or candidate < int(max_batch_size):
                next_candidate = candidate * 2
                if max_batch_size is not None:
                    next_candidate = min(next_candidate, int(max_batch_size))
                if next_candidate <= candidate:
                    break

                result = _probe(next_candidate)
                if result["fits"]:
                    best_probe = result
                    low_good = int(next_candidate)
                    candidate = int(next_candidate)
                    if max_batch_size is not None and candidate >= int(max_batch_size):
                        break
                else:
                    high_bad = int(next_candidate)
                    break

            if high_bad is None:
                selected_batch_size = int(low_good)
            else:
                lo, hi = int(low_good), int(high_bad)
                while lo + 1 < hi:
                    mid = (lo + hi) // 2
                    result = _probe(mid)
                    if result["fits"]:
                        best_probe = result
                        lo = mid
                    else:
                        hi = mid
                selected_batch_size = int(lo)
        else:
            lo, hi = 0, int(initial_batch_size)
            while lo + 1 < hi:
                mid = (lo + hi) // 2
                result = _probe(mid)
                if result["fits"]:
                    best_probe = result
                    lo = mid
                else:
                    hi = mid
            selected_batch_size = int(lo)

        selected_batch_size = self._round_down_to_multiple(selected_batch_size, multiple_of)
        if selected_batch_size < int(min_batch_size):
            selected_batch_size = 0

        response: Dict[str, Any] = {
            "selected_batch_size": int(selected_batch_size),
            "mode": str(mode),
            "device": str(self._normalize_cuda_device(device)),
            "probe_history": probe_history,
        }

        if best_probe is not None:
            response["peak_memory_bytes"] = int(best_probe["peak_memory_bytes"])
            response["peak_memory_gb"] = float(best_probe["peak_memory_bytes"]) / (1024 ** 3)

        if target_effective_batch_size is not None and selected_batch_size > 0:
            response["target_effective_batch_size"] = int(target_effective_batch_size)
            response["gradient_accumulation_steps"] = int(
                math.ceil(int(target_effective_batch_size) / int(selected_batch_size))
            )

        if model_ref is not None:
            response["model_ref"] = str(model_ref)
        if actLs is not None:
            response["seq_len_bp"] = int(actLs)

        return response
