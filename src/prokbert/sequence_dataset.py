import typing as t

import os
import time
import bisect
import logging

import torch
import datasets
import pandas as pd

from prokbert import helper
from prokbert.types import (
    Contig,
    ContigMetaData,
    Orientation,
    Sequence,
    SequenceId,
    SequenceInterval,
)
from prokbert.constants import RC_TABLE, FORWARD, BACKWARD


logger = logging.getLogger(__name__)


class SequenceDataset(object):
    def __init__(self) -> None:
        self.sequence: Sequence = ""
        self.metadata: t.List[ContigMetaData] = []
        self._starts: t.List[int] = []

    def load_contigs(self, file_paths: t.List[str]) -> t.List[Contig]:
        data = []
        for file_path in file_paths:
            contigs = self.load_contig(file_path)
            data.extend(contigs)
        return data

    def load_contig(self, file_path: str) -> t.List[Contig]:
        data = []
        contigs = helper.load_file(file_path)
        for contig in contigs:
            contig_forward = self._create_contig(contig, reverse_complement=False)
            data.append(contig_forward)
        return data

    def _create_contig(self, contig, reverse_complement: bool = True) -> Contig:
        orientation: Orientation = BACKWARD if reverse_complement else FORWARD
        seq = contig.seq.reverse_complement() if reverse_complement else contig.seq
        return {
            "contig_id": contig.id,
            "sequence": str(seq).upper(),
            "orientation": orientation,
            "description": contig.description,
        }

    @staticmethod
    def convert_to(
        data: t.List[Contig],
        return_as: str = "list"
    ) -> t.Union[t.List[Contig], pd.DataFrame, datasets.Dataset]:
        if return_as == "list":
            return data
        elif return_as == "pandas":
            return pd.DataFrame(data)
        elif return_as == "datasets":
            return datasets.Dataset.from_list(data)
        else:
            raise ValueError(
                    f"Invalid value for return_as: {return_as}. "
                    f"Supported values are 'list', 'pandas', 'datasets'."
                )

    def create_dataset(self, file_paths: t.List[str], save_dir: str | None = None) -> None:

        t0 = time.perf_counter()

        dataset = self.load_contigs(file_paths)

        offset = 0
        sequences: t.List[Sequence] = []
        metadata: t.List[ContigMetaData] = []

        for i, record in enumerate(dataset):
            end = offset + len(record["sequence"])
            metadata.append({
                "contig_id": record["contig_id"],
                "sequence_id": i,
                "coordinate": (offset, end),
                "orientation": record["orientation"],
                "description": record["description"],
            })
            sequences.append(record["sequence"])
            offset = end

        self.sequence = "".join(sequences)
        self.metadata = metadata
        self._starts = [contig["coordinate"][0] for contig in self.metadata]

        logging.info(f"Dataset created in {time.perf_counter() - t0:.2f} seconds.")

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

            helper.save_sequence(self.sequence, save_dir, "sequence.npy")
            helper.save_json(self.metadata, save_dir, "metadata.json")

            logging.info(f"Dataset saved to '{save_dir}'.")

    def load_dataset(self, dir_path: str, to_string: bool = False) -> None:

        self.sequence = helper.load_sequence(os.path.join(dir_path, "sequence.npy"), to_string=to_string)
        self.metadata = helper.read_json(os.path.join(dir_path, "metadata.json"))
        self._starts = [contig["coordinate"][0] for contig in self.metadata]

        logging.info(f"Dataset loaded from '{dir_path}'.")

    def get_coordinates_from_sequence_id(self, sequence_id: int) -> SequenceInterval:

        if sequence_id >= len(self.metadata) or sequence_id < 0:
            raise ValueError(f"Sequence ID {sequence_id} is out of bounds.")

        return self.metadata[sequence_id]["coordinate"]

    def get_sequence_id_from_start_coordinate(self, position: int) -> SequenceId:
        # O(log n) using binary search

        i = bisect.bisect_right(self._starts, position) - 1
        if i >= 0:
            start, end = self.metadata[i]["coordinate"]
            if start <= position < end:
                return self.metadata[i]["sequence_id"]

        raise ValueError(f"Coordinate {position} not found in metadata.")

    def get_sequence_from_metadata(
        self,
        sequence_id: int,
        start_coor: int,
        end_coor: int,
        orientation: Orientation
    ) -> Sequence:

        # the (start, end) coordinates are saves as cumulative coordinates in the concatenated sequence (self.sequence)
        start, end = self.get_coordinates_from_sequence_id(sequence_id)
        s, e = start + start_coor, start + end_coor

        if start_coor >= end_coor:
            raise ValueError(f"Start coordinate {start_coor} must be less than end coordinate {end_coor}.")
        if start_coor < 0 or end_coor > end - start:
            raise ValueError(
                f"Coordinates {start_coor}-{end_coor} are out of bounds for "
                f"sequence ID {sequence_id} with sequence length {end - start}. "
                f"(The coordinates were mapped to {s}-{e} with full sequence length {len(self.sequence)}.)"
            )

        seq = self.sequence[s:e]
        if orientation == FORWARD:
            return seq
        elif orientation == BACKWARD:
            return self.reverse_complement(seq)
        else:
            raise ValueError(f"Invalid orientation: {orientation}. Must be '{FORWARD}' or '{BACKWARD}'.")

    def reverse_complement(self, sequence: Sequence) -> str: # revcomp from https://github.com/nbrg-ppcu/prokbert/blob/development/src/prokbert/sequtils.py
        return sequence.translate(RC_TABLE)[::-1] # translate then reverse


class EmbeddingDataset(object):
    def __init__(
        self,
        sequence_dataset: SequenceDataset,
        config_path: str,
        embedding_file: str,
    ) -> None:
        super().__init__()

        self.sequence_dataset = sequence_dataset
        self.config = self.load_config(config_path)
        # tokenized len of self.sequence_dataset.sequence // pooling_length x dim (prokbert-mini-long: 384, mini2-c: 1024)
        self.embedding = torch.load(
            embedding_file, map_location="cpu", mmap=True, weights_only=True
        )

    def load_config(self, config_path: str) -> dict:
        config = helper.load_yaml(config_path)
        return {
            "dataset": config["dataset"], # e.g. huggingface dataset name
            "tokenizer": config["tokenizer"],
            "kmer": config["kmer"],
            "shift": config["shift"],
            "model": config["model"],
            "pooling_strategy": config["pooling_strategy"],
            "pooling_length": config["pooling_length"],
            "special_tokens": config["special_tokens"], # e.g. (CLS, EOS)
        }

    def get_embedding_from_sequence_id_with_coordinates(
        self,
        sequence_id: int,
        start_coor: int,
        end_coor: int,
    ) -> torch.Tensor:
        # (start_coor, end_coor) is relative to the contig / sequence,
        # not the concatenated sequence, which are cumulative coordinates

        start_coor_seq, end_coor_seq = self.sequence_dataset.get_coordinates_from_sequence_id(sequence_id)
        if (start_coor < 0 or end_coor < start_coor) or end_coor > (end_coor_seq - start_coor_seq):
            raise ValueError(
                f"Coordinates {start_coor}-{end_coor} are out of bounds for "
                f"sequence ID {sequence_id} with sequence length {end_coor_seq - start_coor_seq}."
            )

        logging.debug(f"Sequence ID {sequence_id} with requested coordinates {start_coor}-{end_coor}.")

        start_coor_in_tokens = self.calculate_lca_num_tokens(
            start_coor + start_coor_seq,
            self.config["kmer"],
            self.config["shift"],
            special_tokens=len(self.config["special_tokens"])
        )
        end_coor_in_tokens  = self.calculate_lca_num_tokens(
            end_coor + start_coor_seq,
            self.config["kmer"],
            self.config["shift"],
            special_tokens=len(self.config["special_tokens"])
        )
        logging.debug(
            f"Start token: {start_coor_in_tokens}, End token: {end_coor_in_tokens} "
            f"(in tokenised sequence with kmer={self.config['kmer']} and shift={self.config['shift']})"
        )

        start = start_coor_in_tokens // self.config["pooling_length"]
        end  = end_coor_in_tokens // self.config["pooling_length"]

        logging.debug(f"Start index in embedding: {start}, End index in embedding: {end} with embedding shape {self.embedding.shape}")

        # safaty check to ensure the coordinates are within the bounds of the embedding tensor
        if start < 0 or end > self.embedding.shape[0] or start >= end:
            raise ValueError(
                f"Coordinates {start}-{end} are out of bounds for sequence ID {sequence_id} "
                f"with embedding shape {self.embedding.shape}. Please note that these coordinates are "
                "calculated based on the kmer and shift parameters, i.e. the tokenised sequences,"
                "and may not directly correspond to the original (untokenised) sequence coordinates."
            )
        return self.embedding[start:end]

    @staticmethod
    def calculate_lca_num_tokens(
        seq_len: int,
        kmer: int,
        shift: int,
        special_tokens: int = 2 # e.g. (CLS, EOS)
    ) -> int:
        """
        Number of LCA tokens for a single segment (offset 0).

        K-mers start at positions 0, shift, 2*shift, ... and must fit entirely
        in the sequence, giving floor((seq_len - kmer) / shift) + 1 k-mers.
        Special tokens are added once per segment. For a whole contig split
        into segments, sum this over the segments.

        Derived from the LCA tokenization in ProkBERT (Ligeti et al., 2024, Sec. 2.1.1)
        and matches lca_tokenize_segment in prokbert/sequtils.py for offset 0.

        Args:
            seq_len: Length of the segment in nucleotides.
            kmer: K-mer size.
            shift: Step between consecutive k-mer start positions.
            special_tokens: Number of special tokens added per segment.

        Returns:
            Number of tokens, including special tokens.

        Example:
            'AAGTCCAGGATC' (length 12), kmer=6, shift=2 gives the k-mers AAGTCC, GTCCAG, CCAGGA, AGGATC:

            >>> SequenceDataset.calculate_lca_num_tokens(12, 6, 2, special_tokens=0)
            4
            >>> SequenceDataset.calculate_lca_num_tokens(12, 6, 2)
            6
        """
        # see 2 Materials and methods in https://pmc.ncbi.nlm.nih.gov/articles/PMC10810988/
        if seq_len < kmer:
            return special_tokens
        return (seq_len - kmer) // shift + 1 + special_tokens

