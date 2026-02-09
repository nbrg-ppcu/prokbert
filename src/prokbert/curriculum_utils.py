# Some helper function

import os

import numpy as np
import pandas as pd
import torch
import umap.umap_ as umap
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import silhouette_score, pairwise_distances

def get_embedding(model,
                  dataset,
                  data_collator,
                  device,
                  batch_size=128,
                  columns_to_keep = ['input_ids', 'attention_mask'],):

    """
    Get embeddings for ProkBERT like models using batch inferense
    """
    loader = DataLoader(
        dataset.remove_columns([col for col in dataset.column_names if col not in columns_to_keep]),
        batch_size=batch_size,
        shuffle=False,          # critical: preserve dataset order
        collate_fn=data_collator,
    )

    reps = []
    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k in ("input_ids", "attention_mask")}

            out = model(**inputs)

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

    embeddings = np.concatenate(reps, axis=0)

    return embeddings

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

    embeddings = get_embedding(
        model=model,
        dataset=dataset,
        data_collator=data_collator,
        device=device,
        batch_size=batch_size,
    )

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

def evaluate_embeddings(embeddings, labels):
    """
    Evaluate embedding quality using silhouette score based on cosine distance.
    """
    D = pairwise_distances(embeddings, metric='cosine')
    score = silhouette_score(D, labels, metric='precomputed')
    return float(score)