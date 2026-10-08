from typing import Iterator

import random
import itertools

import numpy as np
from torch.utils.data import IterableDataset, get_worker_info

from prokbert.types import SegmentId, Segment, ContigMetaData, SequenceId
from prokbert.sequence_dataset import SequenceDataset


__all__ = ["RandomSegmentDataset", "ContiguousSegmentDataset"]


class SegmentDataset(IterableDataset):
    def __init__(
        self,
        sequence_dataset: SequenceDataset,
        min_segment_length: int,
        max_segment_length: int,
        return_sequence: bool = False
    ) -> None:
        super().__init__()

        if not 0 < min_segment_length <= max_segment_length:
            raise ValueError(
                f"Expected 0 < min_segment_length <= max_segment_length, "
                f"got {min_segment_length} and {max_segment_length}."
            )
        self.min_segment_length = min_segment_length
        self.max_segment_length = max_segment_length
        self.sequence_dataset = sequence_dataset
        self.return_sequence = return_sequence

    def create_segment(
        self,
        segment_id: SegmentId,
        contig: ContigMetaData,
        coor_abs_start: int,
        coor_abs_end: int
    ) -> Segment:
        contig_coor_abs_start, _ = contig["coordinate"]
        coor_rel_start = coor_abs_start - contig_coor_abs_start
        coor_rel_end = coor_abs_end - contig_coor_abs_start
        segment =  Segment(
            segment_id=segment_id,
            contig_id=contig["contig_id"],
            genome_id=contig.get("genome_id"),
            sequence_id=contig["sequence_id"],
            absolute_coordinate=(coor_abs_start, coor_abs_end),
            relative_coordinate=(coor_rel_start, coor_rel_end),
            orientation=contig["orientation"],
            label=contig.get("label"),
        )
        if self.return_sequence:
            segment["sequence"] = self.read_sequence(coor_abs_start, coor_abs_end)
        return segment

    def read_sequence(self, segment_coor_abs_start: int, segment_coor_abs_end: int) -> str:
        sequence = self.sequence_dataset.get_sequence_by_absolute_coordinates(
            segment_coor_abs_start, segment_coor_abs_end
        )
        return sequence.tobytes().decode("ascii") if isinstance(sequence, np.ndarray) else sequence


class RandomSegmentDataset(SegmentDataset):

    def __iter__(self) -> Iterator[Segment | str]:

        worker = get_worker_info()
        worker_id, num_workers = (worker.id, worker.num_workers) if worker is not None else (0, 1)

        segment_id = 0
        total_len = self.sequence_dataset.get_sequence_len()

        while True:
            pos = random.randrange(0, total_len) # left closed, right open interval

            sequence_id = self.sequence_dataset.get_sequence_id_from_start_coordinate(pos)
            contig = self.sequence_dataset.get_contig_metadata_from_sequence_id(sequence_id)
            contig_coor_abs_start, contig_coor_abs_end = contig["coordinate"]
            contig_len = contig_coor_abs_end - contig_coor_abs_start

            if contig_len < self.min_segment_length:
                continue # too short, draw again
            if contig_len <= self.max_segment_length: # whole contig
                segment_coor_abs_start = contig_coor_abs_start
                segment_coor_abs_end = contig_coor_abs_end
            else:
                segment_coor_abs_start = random.randint( # inclusive
                    contig_coor_abs_start, contig_coor_abs_end - self.max_segment_length
                )
                segment_coor_abs_end = segment_coor_abs_start + self.max_segment_length

            yield self.create_segment(
                segment_id * num_workers + worker_id, # workers gets different segment_ids
                contig,
                segment_coor_abs_start,
                segment_coor_abs_end,
            )
            segment_id += 1


# TODO shuffling is not implemented. Do we need it here?
class ContiguousSegmentDataset(SegmentDataset):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._first_segment_ids = self._calculate_segment_ids()

    def _calculate_segment_ids(self):
        """Get first segment id for each contig. sequence_ids are order in sequence_dataset"""
        counts = []
        for contig in self.sequence_dataset.metadata:
            start, end = contig["coordinate"]
            quotient, remainder = divmod(end - start, self.max_segment_length)
            n_segments = quotient + (remainder >= self.min_segment_length)
            counts.append(n_segments)
        return [0, *itertools.accumulate(counts)]

    def create_segments_from_contig(self, sequence_id: SequenceId) -> list[Segment]:
        """All segments of one contig, in order."""

        contig = self.sequence_dataset.get_contig_metadata_from_sequence_id(sequence_id)
        contig_coor_abs_start, contig_coor_abs_end = contig["coordinate"]
        first_segment_id = self._first_segment_ids[sequence_id]

        segments = []
        for segment_coor_abs_start in range(contig_coor_abs_start, contig_coor_abs_end, self.max_segment_length):
            segment_coor_abs_end = min(segment_coor_abs_start + self.max_segment_length, contig_coor_abs_end)

            if segment_coor_abs_end - segment_coor_abs_start < self.min_segment_length:
                continue # skip the too-short contig, i.e. last slice

            segments.append(
                self.create_segment(
                    first_segment_id + len(segments),
                    contig,
                    segment_coor_abs_start,
                    segment_coor_abs_end,
                )
            )
        return segments

    def __iter__(self) -> Iterator[Segment | str]:
        contigs = self.sequence_dataset.metadata
        worker = get_worker_info()
        if worker is not None: # each worker takes every num_workers-th contig
            contigs = contigs[worker.id::worker.num_workers]

        for contig in contigs:
            yield from self.create_segments_from_contig(contig["sequence_id"])
