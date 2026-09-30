import typing as t

Id = t.Union[int, str]
GenomeId = Id
ContigId = Id
SequenceId = Id
SegmentId = Id

Sequence = t.AnyStr
Segment = t.AnyStr
Segments = t.List[Sequence]
Orientation = t.Literal["forward", "backward"]
SequenceInterval = t.Tuple[int, int] # (start, end) coordinate pair
Description = t.Optional[str]

class Contig(t.TypedDict):
    """A contig as read from the file, before building the dataset."""
    contig_id: ContigId
    sequence: Sequence
    orientation: Orientation
    description: Description


class ContigMetaData(t.TypedDict):
    """Metadata for a contig inside the concatenated sequence."""
    contig_id: ContigId
    sequence_id: SequenceId
    coordinate: SequenceInterval
    orientation: Orientation
    description: Description
