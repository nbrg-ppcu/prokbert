# Some helper function

import os

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import umap
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm



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
    plt.suptitle(f'Visualization of embeddings')
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

    grouped_means = df.groupby('group').mean().values

    df['labels'] = labels
    labels_pooled = df[['labels', 'group']].groupby('group').first().values[:, 0]
    seq_ids_pooled = df.groupby('group')['group'].agg('first').values
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
