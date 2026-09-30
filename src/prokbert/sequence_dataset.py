import typing as t

import os
import time
import bisect

import datasets
import pandas as pd

from src.prokbert import helper
from src.prokbert.types import (
    Contig,
    ContigMetaData,
    Orientation,
    Sequence,
    SequenceId,
    SequenceInterval,
)
from src.prokbert.constants import RC_TABLE


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
        orientation: Orientation = "backward" if reverse_complement else "forward"
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

        print(f"Dataset created in {time.perf_counter() - t0:.2f} seconds.")

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

            helper.save_sequence(self.sequence, save_dir, "sequence.npy")
            helper.save_json(self.metadata, save_dir, "metadata.json")

            print(f"Dataset saved to '{save_dir}'.")

    def load_dataset(self, dir_path: str) -> None:

        self.sequence = helper.load_sequence(os.path.join(dir_path, "sequence.npy"))
        self.metadata = helper.read_json(os.path.join(dir_path, "metadata.json"))
        self._starts = [contig["coordinate"][0] for contig in self.metadata]

        print(f"Dataset loaded from '{dir_path}'.")

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
        start, _ = self.get_coordinates_from_sequence_id(sequence_id)
        start_coor, end_coor = start + start_coor, start + end_coor

        if start_coor >= end_coor:
            raise ValueError(f"Start coordinate {start_coor} must be less than end coordinate {end_coor}.")
        if start_coor < 0 or end_coor > len(self.sequence):
            raise ValueError(
                f"Coordinates {start_coor}-{end_coor} are out of bounds for "
                f"sequence ID {sequence_id} with sequence length {len(self.sequence)}.")

        seq = self.sequence[start_coor:end_coor]
        if orientation == "forward":
            return seq
        elif orientation == "backward":
            return self.reverse_complement(seq)
        else:
            raise ValueError(f"Invalid orientation: {orientation}. Must be 'forward' or 'backward'.")

    def reverse_complement(self, sequence: Sequence) -> str: # revcomp from https://github.com/nbrg-ppcu/prokbert/blob/development/src/prokbert/sequtils.py
        return sequence.translate(RC_TABLE)[::-1] # translate then reverse

