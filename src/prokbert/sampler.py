import typing as t

import random

from prokbert.constants import CONTIGUOUS, RANDOM
from prokbert.types import Segment, SegmentationType, ContigMetaData
from prokbert.sequence_dataset import SequenceDataset

class Sampler(object):
    def __init__(
        self,
        sequence_dataset: SequenceDataset,
        min_segment_length: int,
        max_length: int,
        segmentation_type: SegmentationType,
        coverage: float = 1.0,
        num_random_segmentation: int = 10
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
        self.num_random_segmentation = num_random_segmentation

    def __len__(self) -> int:
        return len(self.sequence_dataset.metadata)


    def __iter__(self) -> t.Iterator[Segment]:
        if self.segmentation_type == CONTIGUOUS:
            return self.contiguous_segmentation()
        return self.random_segmentation()


    def random_segmentation(self) -> t.Generator[Segment, int, None]:
        for _ in range(self.num_random_segmentation):

            total_len = self.sequence_dataset.get_sequence_len()
            pos = random.randrange(0, total_len) # left closed, right open interval

            seq_id = self.sequence_dataset.get_sequence_id_from_start_coordinate(pos)
            contig = self.sequence_dataset.get_contig_metadata_from_sequence_id(seq_id)
            contig_start, contig_end = contig["coordinate"] # absolute coordinates
            contig_len = contig_end - contig_start # absolute end - star

            if contig_len < self.min_length:
                start, end = contig["coordinate"]
            else:
                start = random.randint(contig_start, contig_end - self.max_length)
                # start can be negative, if contig_end is less then max_length (but bigger then min_length),
                # e.g. min_length=5, max_length=10, contig_start=0, contig_end=8,
                # then -> start would be -2, corrected to 0 in this case
                if start < 0: start = contig_start
                end = start + self.max_length

            yield Segment(
                segment_id = self._next_segment_id,
                contig_id = contig["contig_id"],
                genome_id = contig.get("genome_id"),
                sequence_id = contig["sequence_id"],
                coordinate  = (start, end), # absolute position
                orientation = contig["orientation"]
                )
            self._next_segment_id += 1

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
                    segment_id = idx,
                    contig_id = contig["contig_id"],
                    genome_id = contig.get("genome_id"),
                    sequence_id = contig["sequence_id"],
                    coordinate = (segment_start, segment_end),
                    orientation = contig["orientation"]
                )

