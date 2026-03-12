import os

import umap
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import silhouette_score, pairwise_distances

import torch.nn.functional as F



def deprcompute_umap_for_dataset(
    model,
    dataset,
    data_collator,
    batch_size: int = 64,
    seed: int = 42,
    umap_n_neighbors: int | None = None,
    umap_min_dist: float = 0.001,
    umap_metric: str = "cosine",
    n_components: int = 2,
    device: str | torch.device | None = None,
):
    """
    Compute UMAP coordinates for `dataset` in the *same order* as dataset rows.

    Assumptions:
      - `dataset` is already tokenized and contains `input_ids` and `attention_mask`
      - `data_collator` creates torch tensors for those fields
      - `model(**batch)` returns either:
          * a tensor of shape [B, D] (preferred, like your current utils), or
          * a HF output with `.pooler_output`, or
          * a HF output with `.last_hidden_state` (we use CLS token embedding)

    Returns:
      coords: np.ndarray of shape [N, 2], aligned with dataset order (index 0..N-1)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    was_training = model.training
    model.eval()
    model.to(device)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,          # critical: preserve dataset order
        collate_fn=data_collator,
    )

    reps = []
    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k in ("input_ids", "attention_mask")}
            out = model(**inputs, return_embeddings=True)

            print(out)
            print(inputs)

            # Robust extraction of [B, D] embeddings
            if torch.is_tensor(out):
                emb = out
            elif isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
                emb = out[0]
            elif hasattr(out, "pooler_output") and torch.is_tensor(out.pooler_output):
                emb = out.pooler_output
            elif hasattr(out, "last_hidden_state") and torch.is_tensor(out.last_hidden_state):
                emb = out.last_hidden_state[:, 0, :]  # CLS token
            else:
                raise TypeError(
                    "Model output type not supported for embedding extraction. "
                    "Expected tensor / (tensor, ...) / pooler_output / last_hidden_state."
                )
            reps.append(emb.detach().cpu().float().numpy())
    if was_training:
        model.train()

    embeddings = np.concatenate(reps, axis=0)  # order preserved by DataLoader(shuffle=False)

    n = embeddings.shape[0]
    if umap_n_neighbors is None:
        # close to your current heuristic, but with a sane lower bound
        umap_n_neighbors = max(5, int(np.log(max(n, 2))))

    reducer = umap.UMAP(
        random_state=seed,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        metric=umap_metric,
        n_components=n_components,
        n_jobs=-1,
    )
    coords = reducer.fit_transform(embeddings)  # same row order as `embeddings` -> same as `dataset`
    return embeddings, coords



def compute_umap_for_dataset(
    model,
    dataset,
    data_collator,
    batch_size: int = 64,
    seed: int = 42,
    umap_n_neighbors: int | None = None,
    umap_min_dist: float = 0.001,
    umap_metric: str = "cosine",
    n_components: int = 2,
    device: str | torch.device | None = None,
    normalize_embeddings: bool = True,
):
    """
    Compute embeddings + UMAP coordinates for `dataset` in dataset row order.

    Priority for embedding extraction:
      1) model.encode(...) if available
      2) ProkBert curricular/manual path:
            model.bert(...) -> model._pool_sequence_output(...) -> model.linear(...)
      3) model(..., return_embeddings=True) and read out.embeddings
      4) generic HF fallbacks:
            tensor output / pooler_output / last_hidden_state[:, 0, :]

    Returns
    -------
    embeddings : np.ndarray, shape [N, D]
    coords     : np.ndarray, shape [N, n_components]
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    was_training = model.training
    model.eval()
    model.to(device)

    model_input_keys = {
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "position_ids",
        "inputs_embeds",
    }

    def _collate_only_model_inputs(features):
        trimmed = []
        for f in features:
            trimmed.append({k: v for k, v in f.items() if k in model_input_keys})
        return data_collator(trimmed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_only_model_inputs,
    )

    def _normalize(emb: torch.Tensor) -> torch.Tensor:
        if not normalize_embeddings:
            return emb
        return F.normalize(emb, p=2, dim=1)

    def _extract_embeddings(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        # 1) Preferred explicit encoder API
        if hasattr(model, "encode") and callable(getattr(model, "encode")):
            try:
                emb = model.encode(**inputs, normalize=normalize_embeddings)
            except TypeError:
                emb = model.encode(**inputs)
                emb = _normalize(emb)
            return emb

        # 2) Direct manual path for current ProkBertForCurricularClassification
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

            sequence_output = base_outputs.last_hidden_state
            pooled_output = model._pool_sequence_output(
                sequence_output,
                inputs.get("attention_mask"),
            )

            if hasattr(model, "dropout"):
                pooled_output = model.dropout(pooled_output)

            emb = model.linear(pooled_output)
            return _normalize(emb)

        # 3) Try forward(return_embeddings=True)
        try:
            out = model(**inputs, return_embeddings=True)
        except TypeError:
            out = model(**inputs)

        if hasattr(out, "embeddings") and torch.is_tensor(out.embeddings):
            return _normalize(out.embeddings)

        # 4) Generic fallbacks
        if torch.is_tensor(out):
            emb = out
            if emb.ndim == 3:
                emb = emb[:, 0, :]
            return _normalize(emb)

        if isinstance(out, (tuple, list)) and len(out) == 1 and torch.is_tensor(out[0]):
            emb = out[0]
            if emb.ndim == 3:
                emb = emb[:, 0, :]
            return _normalize(emb)

        if hasattr(out, "pooler_output") and torch.is_tensor(out.pooler_output):
            return _normalize(out.pooler_output)

        if hasattr(out, "last_hidden_state") and torch.is_tensor(out.last_hidden_state):
            return _normalize(out.last_hidden_state[:, 0, :])

        raise TypeError(
            "Could not extract embeddings from model output. "
            "Add a model.encode(...) method or expose embeddings in forward()."
        )

    reps = []
    with torch.no_grad():
        for batch in loader:
            inputs = {
                k: v.to(device)
                for k, v in batch.items()
                if k in model_input_keys and torch.is_tensor(v)
            }
            emb = _extract_embeddings(inputs)
            reps.append(emb.detach().cpu().float().numpy())

    if was_training:
        model.train()

    embeddings = np.concatenate(reps, axis=0)

    n = embeddings.shape[0]
    if umap_n_neighbors is None:
        umap_n_neighbors = max(5, int(np.log(max(n, 2))))

    reducer = umap.UMAP(
        random_state=seed,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        metric=umap_metric,
        n_components=n_components,
        n_jobs=-1,
    )
    coords = reducer.fit_transform(embeddings)
    return embeddings, coords

def evaluate_embeddings(embeddings, labels):
    """
    Evaluate embedding quality using silhouette score based on cosine distance.
    """
    D = pairwise_distances(embeddings, metric='cosine')
    score = silhouette_score(D, labels, metric='precomputed')
    return float(score)

