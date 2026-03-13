from __future__ import annotations

from typing import Callable, Iterable, Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import umap
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader


MODEL_INPUT_KEYS = (
    "input_ids",
    "attention_mask",
    "token_type_ids",
    "position_ids",
    "inputs_embeds",
)


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def build_model_input_collator(
    data_collator: Callable,
    model_input_keys: Sequence[str] = MODEL_INPUT_KEYS,
) -> Callable:
    allowed = set(model_input_keys)

    def _collate(features):
        trimmed = [{k: v for k, v in f.items() if k in allowed} for f in features]
        return data_collator(trimmed)

    return _collate


def _maybe_normalize(embeddings: torch.Tensor, normalize: bool) -> torch.Tensor:
    if not normalize:
        return embeddings
    return F.normalize(embeddings, p=2, dim=1)


def extract_batch_embeddings(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    *,
    normalize_embeddings: bool = True,
) -> torch.Tensor:
    """
    Model-agnostic batch embedding extraction.

    Priority:
      1) model.encode(...)
      2) model(..., return_embeddings=True) -> out.embeddings
      3) ProkBert manual path (bert + pooling + linear)
      4) generic HF fallbacks
    """

    # 1) Preferred explicit model API
    if hasattr(model, "encode") and callable(getattr(model, "encode")):
        try:
            return model.encode(**inputs, normalize=normalize_embeddings)
        except TypeError:
            emb = model.encode(**inputs)
            return _maybe_normalize(emb, normalize_embeddings)

    # 2) forward(..., return_embeddings=True)
    try:
        out = model(
            **inputs,
            return_embeddings=True,
            normalize_embeddings=normalize_embeddings,
        )
        if hasattr(out, "embeddings") and torch.is_tensor(out.embeddings):
            return out.embeddings
    except TypeError:
        pass

    # 3) Current ProkBert fallback
    if (
        hasattr(model, "bert")
        and hasattr(model, "_pool_sequence_output")
        and hasattr(model, "linear")
    ):
        base_outputs = model.bert(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
            token_type_ids=inputs.get("token_type_ids"),
            position_ids=inputs.get("position_ids"),
            inputs_embeds=inputs.get("inputs_embeds"),
            return_dict=True,
        )
        pooled_output = model._pool_sequence_output(
            base_outputs.last_hidden_state,
            inputs.get("attention_mask"),
        )
        emb = model.linear(pooled_output)
        return _maybe_normalize(emb, normalize_embeddings)

    # 4) Generic fallbacks
    out = model(**inputs)

    if torch.is_tensor(out):
        emb = out
        if emb.ndim == 3:
            emb = emb[:, 0, :]
        return _maybe_normalize(emb, normalize_embeddings)

    if isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
        emb = out[0]
        if emb.ndim == 3:
            emb = emb[:, 0, :]
        return _maybe_normalize(emb, normalize_embeddings)

    if hasattr(out, "pooler_output") and torch.is_tensor(out.pooler_output):
        return _maybe_normalize(out.pooler_output, normalize_embeddings)

    if hasattr(out, "last_hidden_state") and torch.is_tensor(out.last_hidden_state):
        return _maybe_normalize(out.last_hidden_state[:, 0, :], normalize_embeddings)

    raise TypeError(
        "Could not extract embeddings from model output. "
        "Implement model.encode(...) or expose embeddings in forward()."
    )


def iter_embeddings_from_dataloader(
    model: torch.nn.Module,
    dataloader: DataLoader,
    *,
    device: str | torch.device | None = None,
    normalize_embeddings: bool = True,
    model_input_keys: Sequence[str] = MODEL_INPUT_KEYS,
    move_model_to_device: bool = True,
) -> Iterator[torch.Tensor]:
    """
    Yield embedding batches as torch.Tensor on CPU.
    """
    device = resolve_device(device)
    keyset = set(model_input_keys)

    was_training = model.training
    if move_model_to_device:
        model.to(device)
    model.eval()

    try:
        with torch.inference_mode():
            for batch in dataloader:
                inputs = {
                    k: v.to(device, non_blocking=True)
                    for k, v in batch.items()
                    if k in keyset and torch.is_tensor(v)
                }
                emb = extract_batch_embeddings(
                    model,
                    inputs,
                    normalize_embeddings=normalize_embeddings,
                )
                yield emb.detach().cpu()
    finally:
        if was_training:
            model.train()


def compute_embeddings_for_dataloader(
    model: torch.nn.Module,
    dataloader: DataLoader,
    *,
    device: str | torch.device | None = None,
    normalize_embeddings: bool = True,
    dtype: np.dtype = np.float32,
    memmap_path: str | None = None,
    expected_rows: int | None = None,
    model_input_keys: Sequence[str] = MODEL_INPUT_KEYS,
    move_model_to_device: bool = True,
) -> np.ndarray:
    """
    Materialize embeddings for an existing dataloader.

    If memmap_path is provided, embeddings are streamed into a NumPy memmap.
    """
    storage = None
    chunks: list[np.ndarray] = []
    offset = 0

    for emb in iter_embeddings_from_dataloader(
        model=model,
        dataloader=dataloader,
        device=device,
        normalize_embeddings=normalize_embeddings,
        model_input_keys=model_input_keys,
        move_model_to_device=move_model_to_device,
    ):
        arr = emb.to(torch.float32).numpy().astype(dtype, copy=False)

        if memmap_path is None:
            chunks.append(arr)
            continue

        if storage is None:
            if expected_rows is None:
                raise ValueError("expected_rows must be set when using memmap_path.")
            storage = np.memmap(
                memmap_path,
                mode="w+",
                dtype=dtype,
                shape=(expected_rows, arr.shape[1]),
            )

        next_offset = offset + arr.shape[0]
        storage[offset:next_offset] = arr
        offset = next_offset

    if memmap_path is not None:
        if storage is None:
            raise ValueError("No embeddings were produced.")
        return storage[:offset] if offset != storage.shape[0] else storage

    if not chunks:
        return np.empty((0, 0), dtype=dtype)

    return np.concatenate(chunks, axis=0)


def compute_embeddings_for_dataset(
    model: torch.nn.Module,
    dataset,
    data_collator: Callable,
    *,
    batch_size: int = 64,
    device: str | torch.device | None = None,
    normalize_embeddings: bool = True,
    model_input_keys: Sequence[str] = MODEL_INPUT_KEYS,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    dtype: np.dtype = np.float32,
    memmap_path: str | None = None,
) -> np.ndarray:
    """
    Compute embeddings for a dataset in dataset row order.
    """
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=build_model_input_collator(
            data_collator,
            model_input_keys=model_input_keys,
        ),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return compute_embeddings_for_dataloader(
        model=model,
        dataloader=loader,
        device=device,
        normalize_embeddings=normalize_embeddings,
        dtype=dtype,
        memmap_path=memmap_path,
        expected_rows=len(dataset),
        model_input_keys=model_input_keys,
    )


def fit_umap(
    embeddings: np.ndarray,
    *,
    seed: int = 42,
    umap_n_neighbors: int | None = None,
    umap_min_dist: float = 0.001,
    umap_metric: str = "cosine",
    n_components: int = 2,
    reducer_factory=None,
    reducer_kwargs: dict | None = None,
):
    """
    Fit a UMAP reducer and return (reducer, coords).
    """
    embeddings = np.asarray(embeddings)
    if embeddings.ndim != 2:
        raise ValueError(f"`embeddings` must be 2D, got shape={embeddings.shape}")

    n = embeddings.shape[0]
    if n == 0:
        raise ValueError("Cannot fit UMAP on an empty embedding array.")

    if umap_n_neighbors is None:
        umap_n_neighbors = max(5, int(np.log(max(n, 2))))

    if reducer_factory is None:
        reducer_factory = umap.UMAP

    reducer_kwargs = {} if reducer_kwargs is None else dict(reducer_kwargs)

    reducer = reducer_factory(
        random_state=seed,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        metric=umap_metric,
        n_components=n_components,
        **reducer_kwargs,
    )
    coords = reducer.fit_transform(embeddings)

    if torch.is_tensor(coords):
        coords = coords.detach().cpu().numpy()
    elif hasattr(coords, "get"):  # CuPy / cuML
        coords = coords.get()
    else:
        coords = np.asarray(coords)

    return reducer, coords


def transform_umap(reducer, embeddings: np.ndarray) -> np.ndarray:
    """
    Project embeddings with an already-fitted reducer.
    """
    coords = reducer.transform(embeddings)

    if torch.is_tensor(coords):
        coords = coords.detach().cpu().numpy()
    elif hasattr(coords, "get"):
        coords = coords.get()
    else:
        coords = np.asarray(coords)

    return coords


def compute_umap_for_dataset(
    model,
    dataset,
    data_collator,
    *,
    batch_size: int = 64,
    seed: int = 42,
    umap_n_neighbors: int | None = None,
    umap_min_dist: float = 0.001,
    umap_metric: str = "cosine",
    n_components: int = 2,
    device: str | torch.device | None = None,
    normalize_embeddings: bool = True,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    memmap_path: str | None = None,
    return_reducer: bool = False,
    reducer_factory=None,
    reducer_kwargs: dict | None = None,
):
    """
    Backward-compatible convenience wrapper.
    """
    embeddings = compute_embeddings_for_dataset(
        model=model,
        dataset=dataset,
        data_collator=data_collator,
        batch_size=batch_size,
        device=device,
        normalize_embeddings=normalize_embeddings,
        num_workers=num_workers,
        pin_memory=pin_memory,
        memmap_path=memmap_path,
    )

    reducer, coords = fit_umap(
        embeddings=embeddings,
        seed=seed,
        umap_n_neighbors=umap_n_neighbors,
        umap_min_dist=umap_min_dist,
        umap_metric=umap_metric,
        n_components=n_components,
        reducer_factory=reducer_factory,
        reducer_kwargs=reducer_kwargs,
    )

    if return_reducer:
        return embeddings, coords, reducer
    return embeddings, coords


def evaluate_embeddings(
    embeddings,
    labels,
    *,
    metric: str = "cosine",
    sample_size: int | None = None,
    random_state: int = 42,
) -> float:
    """
    Silhouette score directly on embeddings.

    Notes
    -----
    - Removes the explicit NxN distance matrix allocation from the old code.
    - For very large datasets, set sample_size.
    """
    embeddings = np.asarray(embeddings)
    labels = np.asarray(labels)

    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Mismatched lengths: {embeddings.shape[0]} embeddings vs {labels.shape[0]} labels"
        )

    if np.unique(labels).size < 2:
        return float("nan")

    score = silhouette_score(
        embeddings,
        labels,
        metric=metric,
        sample_size=sample_size,
        random_state=random_state,
    )
    return float(score)