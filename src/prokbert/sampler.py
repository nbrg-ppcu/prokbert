import typing as t
import os
import time
import random
import numpy as np

from prokbert.constants import CONTIGUOUS
from prokbert.types import Segment, SegmentationType
from prokbert.sequence_dataset import SequenceDataset


class Sampler(object):
    def __init__(
        self,
        sequence_dataset: SequenceDataset,
        min_segment_length: int,
        max_length: int,
        segmentation_type: SegmentationType,
        coverage: float = 1.0,
    ) -> None:

        if not 0 < min_segment_length <= max_length:
            raise ValueError(f"Expected 0 < min_length <= max_length, got {min_segment_length} and {max_length}.")
        if segmentation_type not in t.get_args(SegmentationType):
            raise ValueError(
                f"Invalid segmentation_type: {segmentation_type!r}. "
                f"Supported values are {', '.join(map(repr, t.get_args(SegmentationType)))}."
            )
        if coverage <= 0:
            raise ValueError(f"coverage must be positive, got {coverage}.")

        self.min_length = min_segment_length
        self.max_length = max_length
        self.segmentation_type = segmentation_type
        self.coverage = coverage

        self.sequence_dataset = sequence_dataset

        self._next_segment_id = 0

    def __len__(self) -> int:
        return len(self.sequence_dataset.metadata)

    def __iter__(self) -> t.Iterator[Segment]:
        if self.segmentation_type == CONTIGUOUS:
            segment = self.contiguous_segmentation()
        else:
            segment = self.random_segmentation()
        if int(os.environ.get("PROKBERT_PROFILE", 0)) >= 1:
            return self._measure_perf(segment)
        return segment

    def random_segmentation(self) -> t.Generator[Segment, int, None]:

        total_len = self.sequence_dataset.get_sequence_len()

        while True:
            pos = random.randrange(0, total_len) # left closed, right open interval

            seq_id = self.sequence_dataset.get_sequence_id_from_start_coordinate(pos)
            contig = self.sequence_dataset.get_contig_metadata_from_sequence_id(seq_id)
            contig_coor_abs_start, contig_coor_abs_end = contig["coordinate"]
            contig_len = contig_coor_abs_end - contig_coor_abs_start

            if contig_len < self.min_length:
                continue # too short for a segment, draw again
            if contig_len <= self.max_length:
                start, end = contig_coor_abs_start, contig_coor_abs_end # whole contig
            else:
                start = random.randint(contig_coor_abs_start, contig_coor_abs_end - self.max_length) # inclusive
                end = start + self.max_length

            yield Segment(
                segment_id = self._increase_segment_id(),
                contig_id = contig["contig_id"],
                genome_id = contig.get("genome_id"),
                sequence_id = contig["sequence_id"],
                absolute_coordinate  = (start, end),
                relative_coordinate  = (start  - contig_coor_abs_start, end - contig_coor_abs_start),
                orientation = contig["orientation"]
                )

    def _increase_segment_id(self) -> int:
        segment_id = self._next_segment_id
        self._next_segment_id += 1
        return segment_id

    def contiguous_segmentation(self, sequence_id: int | None = None) -> t.Generator[Segment, int, None]:

        if sequence_id is None:
            contigs = self.sequence_dataset.metadata
        else:
            contigs = [self.sequence_dataset.get_contig_metadata_from_sequence_id(sequence_id)]

        for contig in contigs:
            coor_abs_start, coor_abs_end = contig["coordinate"]

            for segment_start in range(coor_abs_start, coor_abs_end, self.max_length):

                segment_end = min(segment_start + self.max_length, coor_abs_end)
                # draw again if segment len is smaller then min_length param
                if segment_end - segment_start < self.min_length:
                    continue

                yield Segment(
                    segment_id = self._increase_segment_id(),
                    contig_id = contig["contig_id"],
                    genome_id = contig.get("genome_id"),
                    sequence_id = contig["sequence_id"],
                    absolute_coordinate  = (segment_start, segment_end),
                    relative_coordinate  = (segment_start - coor_abs_start, segment_end - coor_abs_start),
                    orientation = contig["orientation"]
                )

    def _measure_perf(self, segments) -> t.Generator[Segment | str, int, None]:
        seconds, nbytes, n_segments = 0.0, 0, 0
        return_seq = int(os.environ.get("PROKBERT_PROFILE", 0)) >= 2
        try:
            while True:
                t0 = time.perf_counter()
                segment = next(segments, None)
                if return_seq and segment is not None:
                    seq = self.sequence_dataset.get_sequence_by_absolute_coordinates(
                        start=segment["absolute_coordinate"][0], end=segment["absolute_coordinate"][1]
                    )
                    seq = seq.tobytes().decode("ascii") if isinstance(seq, np.ndarray) else seq
                seconds += time.perf_counter() - t0
                if segment is None:
                    return
                start, end = segment["absolute_coordinate"]
                nbytes += end - start
                n_segments += 1
                yield seq if return_seq else segment
        finally:
            segments.close()
            if n_segments:
                print(
                    f"{self.segmentation_type} segmentation: {n_segments} segments, "
                    f"{nbytes / 1e6:.1f} MB in {seconds:.2f} s -> "
                    f"{nbytes / 1e6 / seconds:.1f} MB/s, {n_segments / seconds:.0f} segments/s"
                )
