import typing as t
import os
import time
import random

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
            contig_start, contig_end = contig["coordinate"] # absolute coordinates
            contig_len = contig_end - contig_start

            if contig_len < self.min_length:
                continue # too short for a segment, draw again
            if contig_len <= self.max_length:
                start, end = contig_start, contig_end # whole contig
            else:
                start = random.randint(contig_start, contig_end - self.max_length) # inclusive
                end = start + self.max_length

            yield Segment(
                segment_id = self._increase_segment_id(),
                contig_id = contig["contig_id"],
                genome_id = contig.get("genome_id"),
                sequence_id = contig["sequence_id"],
                coordinate  = (start, end), # absolute position
                orientation = contig["orientation"]
                )

    def _increase_segment_id(self) -> int:
        segment_id = self._next_segment_id
        self._next_segment_id += 1
        return segment_id

    def contiguous_segmentation(self, sequence_id: int | None = None) -> t.Generator[Segment, int, None]:

        if sequence_id is None: # ez így jó
            contigs = self.sequence_dataset.metadata
        else:
            contigs = [self.sequence_dataset.get_contig_metadata_from_sequence_id(sequence_id)]

        for contig in contigs:
            start, end = contig["coordinate"]
            for idx, segment_start in enumerate(range(start, end, self.max_length)):
                segment_end = min(segment_start + self.max_length, end)
                if segment_end - segment_start < self.min_length:
                    continue
                yield Segment(
                    segment_id = self._increase_segment_id(),
                    contig_id = contig["contig_id"],
                    genome_id = contig.get("genome_id"),
                    sequence_id = contig["sequence_id"],
                    coordinate = (segment_start, segment_end),
                    orientation = contig["orientation"]
                )

    def _measure_perf(self, segments) -> t.Generator[Segment, int, None]:
        seconds, nbytes, n_segments = 0.0, 0, 0
        return_seq = int(os.environ.get("PROKBERT_PROFILE", 0)) >= 2
        try:
            while True:
                t0 = time.perf_counter()
                segment = next(segments)
                if return_seq:
                    seq = self.sequence_dataset.get_sequence_by_absolute_coordinates(
                        start=segment["coordinate"][0], end=segment["coordinate"][1]
                    )
                seconds += time.perf_counter() - t0
                if segment is None:
                    return
                start, end = segment["coordinate"]
                nbytes += end - start
                n_segments += 1
                yield seq if return_seq else segment
        finally:  # runs when the consumer stops early (break, islice)
            segments.close()
            if n_segments:
                print(
                    f"{self.segmentation_type} segmentation: {n_segments} segments, "
                    f"{nbytes / 1e6:.1f} MB in {seconds:.2f} s -> "
                    f"{nbytes / 1e6 / seconds:.1f} MB/s, {n_segments / seconds:.0f} segments/s"
                )
