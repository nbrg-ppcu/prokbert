import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import IterableDataset

from . import streaming_utils


class NCBI_dataset(IterableDataset):
    def __init__(self, basedir, batch_size,  tokenizer_type="k6s2", Ls=2048):
        tokenized_dir = os.path.join(basedir, tokenizer_type)
        sequence_metadata_file = os.path.join(basedir, "metadata_only.feather")

        self.metadata = pd.read_feather(sequence_metadata_file)
        self.accession_assembly_mapping = self.metadata.set_index('accession_id')['assembly_id'].to_dict()

        self.unique_assembly = self.metadata['assembly_id'].unique()
        self.cat2id = {cat: i for i, cat in enumerate(self.unique_assembly)}

        self.store = streaming_utils.ShardedTokenStore(tokenized_dir, dtype=np.uint16, seed=1337, verbose=False)
        self.mask_ids = {0, 1, 2, 3} # PAD, UNK, BOS, EOS from LCA tokenizer
        self.Ls = Ls
        self.batch_size = batch_size


    def __iter__(self):
        batch, cids, starts, L_eff = self.store.draw_batch_windows(L=self.Ls, k=self.batch_size, pad_id=0, group_by_shard=False)
        accessions = [self.store.id_to_key(int(i)) for i in cids]
        assembly = [self.accession_assembly_mapping[i.split("|")[1]] for i in accessions]
        assembly_ind = [self.cat2id[i] for i in assembly]
        attention_mask = np.isin(batch, list(self.mask_ids), invert=True).astype(int)
        yield {
            "input_ids": torch.from_numpy(batch).long(),
            "attention_mask": torch.Tensor(attention_mask).long(),
            "labels": torch.Tensor(assembly_ind).long()
        }