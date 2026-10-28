from typing import Literal
import os
import sys
import time
import math
import bisect
import logging

import torch
import datasets
import numpy as np
import pandas as pd
from Bio.SeqRecord import SeqRecord

from prokbert import utils
from prokbert import helper
from prokbert.types import (
    Contig,
    ContigMetaData,
    Orientation,
    Sequence,
    SequenceId,
    SequenceInterval,
)
from prokbert.constants import RC_TABLE, FORWARD, REVERSE


logger = logging.getLogger(__name__)


# TODO should be renamed the SequenceDatabase, also rename the file to database.py
class SequenceDataset(object):
    def __init__(self) -> None:
        self.sequence: Sequence | np.ndarray = ""
        self.metadata: list[ContigMetaData] = []
        self._starts: list[int] = []

    def load_contigs(self, file_paths: list[str]) -> list[Contig]:
        data = []
        for file_path in file_paths:
            contigs = self.load_contig(file_path)
            data.extend(contigs)
        return data

    def load_contig(self, file_path: str) -> list[Contig]:

        t0 = time.perf_counter()

        contigs = helper.load_file(file_path)

        data = [self._create_contig(contig) if isinstance(contig, SeqRecord) else contig for contig in contigs]

        if utils.profiling_enabled():
            seconds = time.perf_counter() - t0
            mb = sum(utils.get_dict_size(c) for c in data) / 1e6
            print(f"Loaded contigs from {file_path}, {mb:.1f} MB in {seconds:.2f} s -> {mb / seconds:.1f} MB/s")
        return data

    def _create_contig(self, contig) -> Contig:
        return Contig(
            genome_id = None, # TODO later
            contig_id = contig.id,
            sequence = str(contig.seq).upper(),
            orientation = FORWARD,
            description = contig.description,
        )

    def convert_to(
        self,
        format: Literal["pandas", "datasets"]
    ) -> pd.DataFrame | datasets.Dataset:
        return helper.convert_to(self.metadata, format)

    def convert_from(
            self,
            data: pd.DataFrame | datasets.Dataset,
            format: Literal["pandas", "datasets"],
        ):
        self.metadata = helper.convert_from(data, format, return_as="contigmetadata")

    def create_dataset(self, file_paths: list[str], save_dir: str | None = None) -> None:

        nbytes = 0
        t0 = time.perf_counter()

        dataset = self.load_contigs(file_paths)

        offset = 0
        sequences = []
        metadata  = []

        for i, record in enumerate(dataset):
            end = offset + len(record["sequence"])
            metadata.append(ContigMetaData(
                genome_id = record.get("genome_id"),
                contig_id = record["contig_id"],
                sequence_id = i,
                coordinate = (offset, end),
                orientation = record["orientation"],
                description = record.get("description"),
                label = record.get("label"),
            ))
            sequences.append(record["sequence"]) # .encode("ascii")
            offset = end

        self.sequence = "".join(sequences)
        self.metadata = metadata
        self._starts = [contig["coordinate"][0] for contig in self.metadata]

        if utils.profiling_enabled():
            secs = time.perf_counter() - t0
            nbytes = sys.getsizeof(self.sequence) + sum(utils.get_dict_size(m) for m in self.metadata)
            print(f"Created dataset, {nbytes / 1e6:.1f} MB in {secs:.2f} s -> {nbytes / 1e6 / secs:.1f} MB/s")

        if save_dir is not None:
            self.save_dataset(save_dir)

    def save_dataset(self, save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)

        t0 = time.perf_counter()

        helper.save_sequence(self.sequence, save_dir, "sequence.npy")
        helper.save_json(self.metadata, save_dir, "metadata.json")

        if utils.profiling_enabled():
            seconds = time.perf_counter() - t0
            file_sizes = utils.file_size(os.path.join(save_dir, "sequence.npy"), os.path.join(save_dir, "metadata.json"))
            print(
                f"Saved dataset to '{save_dir}' ({file_sizes / 1e6:.1f} MB) in {seconds:.2f} s -> {file_sizes / 1e6 / seconds:.1f} MB/s"
            )

        logging.info(f"Dataset saved to '{save_dir}'.")

    def load_dataset(self, dir_path: str, to_string: bool = False) -> None:

        t0 = time.perf_counter()

        self.sequence = helper.load_sequence(os.path.join(dir_path, "sequence.npy"), to_string=to_string)
        self.metadata = helper.read_json(os.path.join(dir_path, "metadata.json"))
        self._starts = [contig["coordinate"][0] for contig in self.metadata]

        if utils.profiling_enabled():
            seconds = time.perf_counter() - t0
            file_sizes = utils.file_size(os.path.join(dir_path, "sequence.npy"), os.path.join(dir_path, "metadata.json"))
            print(
                f"Loaded dataset from '{dir_path}' ({file_sizes / 1e6:.1f} MB) in {seconds:.2f} s -> {file_sizes / 1e6 / seconds:.1f} MB/s"
            )

    def get_contig_metadata_from_sequence_id(self, sequence_id: SequenceId) -> ContigMetaData:
        if sequence_id >= len(self.metadata) or sequence_id < 0:
            raise ValueError(f"Sequence ID {sequence_id} is out of bounds.")
        return self.metadata[sequence_id]

    def get_coordinates_from_sequence_id(self, sequence_id: SequenceId) -> SequenceInterval:

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
        sequence_id: SequenceId,
        start_coor: int, # start, end are relative to the sequence_id
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
        elif orientation == REVERSE:
            return self.reverse_complement(seq)
        else:
            raise ValueError(f"Invalid orientation: {orientation}. Must be '{FORWARD}' or '{REVERSE}'.")

    def reverse_complement(self, sequence: Sequence) -> str: # revcomp from https://github.com/nbrg-ppcu/prokbert/blob/development/src/prokbert/sequtils.py
        if isinstance(sequence, np.ndarray):
            sequence = sequence.tobytes().decode("ascii")
        return sequence.translate(RC_TABLE)[::-1] # translate then reverse

    def get_contig_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """Absolute start and end coordinates of every contig as arrays, indexed by sequence_id.

        Used for vectorized lookups (e.g. mapping many positions to contigs at once with np.searchsorted).
        """
        sequence_ids = np.array([contig["sequence_id"] for contig in self.metadata])
        if not np.array_equal(sequence_ids, np.arange(len(self.metadata))):
            raise ValueError("Expected sequence_ids 0, 1, 2, ... in metadata order.")

        coordinates = np.array([contig["coordinate"] for contig in self.metadata], dtype=np.int64).reshape(-1, 2)
        return coordinates[:, 0], coordinates[:, 1]

    def get_sequence_len(self) -> int:
        return len(self.sequence) if isinstance(self.sequence, str) else self.sequence.shape[0]

    def get_sequence_by_absolute_coordinates(self, start: int, end: int) -> Sequence:
        """Slice of the concatenated sequence; coordinates are absolute, half-open [start, end)."""
        if not 0 <= start < end <= self.get_sequence_len():
            raise ValueError(
                f"Coordinates {start}-{end} are out of bounds for sequence length {self.get_sequence_len()}."
            )
        return self.sequence[start:end]


# TODO should be renamed the EmbeddingDatabase, also rename the file to database.py
class EmbeddingDataset(object):
    def __init__(self, sequence_dataset, config_path, embedding_file, seq_tokens_file) -> None:
        super().__init__()
        self.sequence_dataset = sequence_dataset
        self.config = self.load_config(config_path)
        self.embedding = torch.load(embedding_file, map_location="cpu", mmap=True, weights_only=True)
        self.seq_tokens = torch.load(seq_tokens_file, map_location="cpu", mmap=True, weights_only=True)

        pool = self.config["pooling_length"]
        # hány sort foglal egy contig: felfelé kerekítünk, mert a részben kitöltött utolsó blokk is egy sor
        self.seq_rows = torch.ceil(self.seq_tokens.double() / pool).long()
        # a contig kezdősora: az előtte lévő contigok sorainak összege
        self.row_offsets = torch.cumsum(self.seq_rows, dim=0) - self.seq_rows

        assert self.seq_rows.sum() == self.embedding.shape[0], (
            f"A tokenszámokból {int(self.seq_rows.sum())} sor jön ki, "
            f"de az embedding {self.embedding.shape[0]} soros"
        )

    def load_config(self, config_path: str) -> dict:
        config = helper.load_yaml(config_path)
        return {
            "tokenizer": config["tokenizer"],
            "kmer": config["kmer"],
            "shift": config["shift"],
            "model": config["model"],
            "pooling_strategy": config["pooling_strategy"],
            "pooling_length": config["pooling_length"],
            "special_tokens": config["special_tokens"], # e.g. (CLS, EOS)
        }

    def get_embedding_from_sequence_id_with_relative_coordinates(
        self,
        sequence_id: int,
        coor_rel_start: int,
        coor_rel_end: int,
    ) -> torch.Tensor:
        """
        Pooled embedding rows overlapping the half-open region [coor_rel_start, coor_rel_end)
        of a contig. Coordinates are relative to the contig, 0-based, like Python slicing.
        """
        coor_abs_start, coor_abs_end = self.sequence_dataset.get_coordinates_from_sequence_id(sequence_id)
        contig_len = coor_abs_end - coor_abs_start

        if coor_rel_start < 0 or coor_rel_end <= coor_rel_start:
            raise ValueError(
                f"Coordinates {coor_rel_start}-{coor_rel_end} are incorrect. "
                f"The start coordinate must be non-negative and strictly less than the end coordinate "
                f"(the region is half-open: [start, end))."
            )
        if coor_rel_end > contig_len:
            raise ValueError(
                f"Coordinates {coor_rel_start}-{coor_rel_end} are incorrect. "
                f"The requested end coordinate {coor_rel_end} exceeds the length of "
                f"sequence ID {sequence_id} ({contig_len})."
            )
        logging.debug(f"Sequence ID {sequence_id}: absolute {coor_abs_start}-{coor_abs_end}, "
                    f"requested relative {coor_rel_start}-{coor_rel_end}")

        kmer = self.config["kmer"]
        shift = self.config["shift"]
        pool = self.config["pooling_length"]

        n_tokens = int(self.seq_tokens[sequence_id])     # number of tokens in the contig
        n_rows   = int(self.seq_rows[sequence_id])       # number of pooled rows of the contig
        offset   = int(self.row_offsets[sequence_id])    # first row of the contig in the embedding tensor

        # hány token van az első olyan token előtt, ami a kért relatív kezdő koordinátát tartalmazza
        # (0-tól indexelve ez egyben ennek a tokennek a kezdő indexe)
        coor_start_in_tokens = self.calculate_lca_num_tokens(coor_rel_start, kmer, shift, special_tokens=0)
        # az első olyan token indexe, amely már a kért szakasz vége után kezdődik
        # (mivel félig nyitott tartomány, ezért ez az utolsó átfedő token indexe + 1)
        # (itt állunk meg: ez a tokentartomány nyitott jobb vége)
        coor_end_in_tokens = min(n_tokens, math.ceil(coor_rel_end / shift)) # round up
        logging.debug(f"Tokens within contig: {coor_start_in_tokens}-{coor_end_in_tokens} (of {n_tokens} tokens)")

        # hány teljes poolozott beágyazásokból álló sor van az első szükséges token előtt
        # (0-tól indexelve ez egyben annak a sornak az indexe, amely az első szükséges tokent tartalmazza)
        start = math.floor(coor_start_in_tokens / pool)

        # az első olyan poolozott beágyazásokból álló sor indexe, amely már a szükséges tokenek után kezdődik
        # (mivel félig nyitott tartomány, ezért ez az utolsó szükséges sor indexe + 1)
        # (felfelé kerekítünk, mert a részben szükséges sort is vissza kell adni: egy sort nem lehet félbevágni)
        # (a min biztonsági háló: legfeljebb a contig sorainak száma)
        end = min(math.ceil(coor_end_in_tokens / pool), n_rows)     # round up: include the partially filled block
        logging.debug(f"Rows within contig: {start}-{end} (of {n_rows}), "
                    f"rows in embedding tensor: {offset + start}-{offset + end} (offset {offset})")

        if start >= end:
            raise ValueError(
                f"No embedding row overlaps the region {coor_rel_start}-{coor_rel_end} of sequence ID {sequence_id}. "
                f"Either the contig is shorter than one k-mer (kmer={kmer}), or the region lies in the "
                f"last few nucleotides that no k-mer covers (shift={shift})."
            )

        return self.embedding[offset + start : offset + end]

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