from typing import Literal, TypedDict, TYPE_CHECKING

import numpy as np


Id = int
GenomeId = Id | None
ContigId = Id
SegmentId = Id
SequenceId = Id

Sequence = str | np.ndarray

Segments = list[str]
Orientation = Literal["forward", "backward"]
SequenceInterval = tuple[int, int] # (start, end) coordinate pair
Description = str | None

SegmentationType = Literal["contiguous", "random"]


class Contig(TypedDict):
    """A contig as read from the file, before building the dataset."""
    genome_id: GenomeId
    contig_id: ContigId
    sequence: Sequence
    orientation: Orientation
    description: Description

class ContigMetaData(TypedDict):
    """Metadata for a contig inside the concatenated sequence."""
    genome_id: GenomeId
    contig_id: ContigId
    sequence_id: SequenceId
    coordinate: SequenceInterval
    orientation: Orientation
    description: Description


class Segment(TypedDict):
    """A segment of a contig. Coordinates are contig-relative, half-open [start, end)."""
    genome_id: GenomeId
    segment_id: SegmentId
    contig_id: ContigId
    sequence_id: SequenceId
    coordinate: SequenceInterval
    orientation: Orientation
