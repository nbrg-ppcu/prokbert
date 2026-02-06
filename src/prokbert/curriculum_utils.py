import os

import umap
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader



def get_model_embeddings_umap(grouped_means, labels, seq_ids, plot_path,
                                plot_name = "umap_embedding_visualization.png",
                                umap_n_neighbors=30,
                                umap_min_dist = 0.001,
                                umap_random_state=42,
                                save_plot=True,
                                plot_point_size=1):
    umap_n_neighbors = int(np.log(grouped_means.shape[0]))
    reducer = umap.UMAP(random_state=umap_random_state, n_neighbors = umap_n_neighbors, min_dist = umap_min_dist, metric='cosine', n_jobs=-1, spread=5, n_components=2)
    umap_embeddings = reducer.fit_transform(grouped_means)

    umap_df = pd.DataFrame(data={'umap_1': umap_embeddings[:, 0], 'umap_2': umap_embeddings[:, 1],
    "labels": labels, "sequence_id": seq_ids})


    plt.figure(figsize=(20, 7))
    g = sns.scatterplot(umap_df, x="umap_1", y="umap_2",hue="labels", palette="tab20", s=plot_point_size)
    plt.subplots_adjust(top=0.9)
    g.get_legend().set_visible(False)
    plt.suptitle('Visualization of embeddings')
    plt.savefig(os.path.join(plot_path, plot_name))
    return umap_df

def create_embeddings(model, dataloader, output_name,  plot_name):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    representations = []
    labels = []
    seq_ids = []
    batch_ind = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            columns_to_keep = ['input_ids', 'attention_mask']
            labels.append(batch["labels"])
            seq_ids.append(batch["sequence_id"])
            batch = {k: v.to(device) for k, v in batch.items() if k in columns_to_keep}
            with torch.no_grad():
                mean_pooled = model(**batch)
                representations_batch = mean_pooled.detach().cpu().float().numpy()
                representations.append(representations_batch)
            batch_ind += 1
    model.train()
    embeddings = np.concatenate(representations)
    labels = np.concatenate(labels)
    seq_ids = np.concatenate(seq_ids)

    df = pd.DataFrame(embeddings)
    df['group'] = seq_ids

    df['labels'] = labels
    umap_df = get_model_embeddings_umap(embeddings, labels, seq_ids,
                                        plot_path = output_name,
                                        plot_point_size=50,
                                        plot_name = plot_name)
    return umap_df


def plot_umap_embeddings(
    model,
    dataset,
    data_collator,
    output_dir: str,
    plot_name: str,
    batch_size: int,
    seed: int,
):
    sample_count = min(1000, len(dataset))
    umap_ds = dataset.shuffle(seed=seed).select(range(sample_count))
    umap_loader = DataLoader(
        umap_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )
    os.makedirs(output_dir, exist_ok=True)
    create_embeddings(model, umap_loader, output_dir, plot_name)



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
    return coords
