from typing import Literal, TypedDict, NotRequired

import numpy as np


GenomeId = str | None
ContigId = str
SegmentId = int
SequenceId = int

Sequence = str | np.ndarray

Segments = list[str]
Orientation = Literal["forward", "reverse"] # vars are not allowed in type expression
SequenceInterval = tuple[int, int] # (start, end) coordinate pair
Description = str | None

SegmentationType = Literal["contiguous", "random"]


class Contig(TypedDict):
    """A contig as read from the file, before building the dataset."""
    genome_id: NotRequired[GenomeId]
    contig_id: ContigId
    sequence: str
    orientation: Orientation
    description: NotRequired[Description]
    label: NotRequired[int | None]

class ContigMetaData(TypedDict):
    """Metadata for a contig inside the concatenated sequence."""

    genome_id: NotRequired[GenomeId]
    contig_id: ContigId
    sequence_id: SequenceId
    coordinate: SequenceInterval
    orientation: Orientation
    description: NotRequired[Description]
    label: NotRequired[int | None]


class Segment(TypedDict):
    """A segment of a contig. Coordinates are contig-relative, half-open [start, end)."""
    genome_id: GenomeId
    segment_id: SegmentId
    contig_id: ContigId
    sequence_id: SequenceId
    absolute_coordinate: SequenceInterval
    relative_coordinate: SequenceInterval
    orientation: Orientation
    sequence: NotRequired[str]
    label: NotRequired[int | None]
