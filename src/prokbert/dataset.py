from typing import Iterator

import itertools

import numpy as np
from torch.utils.data import IterableDataset, get_worker_info

from prokbert.types import SegmentId, Segment, ContigMetaData, SequenceId
from prokbert.sequence_dataset import SequenceDataset, EmbeddingDataset


__all__ = ["RandomSegmentDataset", "ContiguousSegmentDataset"]


class SegmentDataset(IterableDataset):
    def __init__(
        self,
        sequence_dataset: SequenceDataset,
        min_segment_length: int,
        max_segment_length: int,
        embedding_dataset: EmbeddingDataset | None = None,
        return_sequence: bool = False,
        return_embedding: bool = False
    ) -> None:
        super().__init__()

        if not 0 < min_segment_length <= max_segment_length:
            raise ValueError(
                f"Expected 0 < min_segment_length <= max_segment_length, "
                f"got {min_segment_length} and {max_segment_length}."
            )
        if embedding_dataset is None and return_embedding:
            raise ValueError(
                f"Cannot return embedding when embedding_dataset is not provided. "
                f"Got embedding_dataset={embedding_dataset} and return_embedding={return_embedding}."
            )
        self.min_segment_length = min_segment_length
        self.max_segment_length = max_segment_length
        self.sequence_dataset = sequence_dataset
        self.return_sequence = return_sequence
        self.embedding_dataset = embedding_dataset
        self.return_embedding = return_embedding

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
        if self.return_embedding and self.embedding_dataset is not None:
            segment["embedding"] = self.embedding_dataset.get_embedding_from_sequence_id_with_relative_coordinates(
              contig["sequence_id"], coor_rel_start, coor_rel_end
            )
        return segment

    def read_sequence(self, segment_coor_abs_start: int, segment_coor_abs_end: int) -> str:
        sequence = self.sequence_dataset.get_sequence_by_absolute_coordinates(
            segment_coor_abs_start, segment_coor_abs_end
        )
        return sequence.tobytes().decode("ascii") if isinstance(sequence, np.ndarray) else sequence


class RandomSegmentDataset(SegmentDataset):
    """Random segments, sampled the same way as ``sequtils.segment_sequences_random``.

    - ceil(coverage * total_length / max_segment_length) start positions are drawn uniformly
      over the concatenated sequence, then sorted (so contigs are hit proportionally to their length)
    - a segment starts at its sampled position and runs max_segment_length bases or until the
      end of its contig, whichever comes first (segments never cross contigs)
    - segments shorter than min_segment_length are dropped
    - segment_ids are 0, 1, 2, ... over the kept segments, in position order, like
      ``segment_sequences(..., AsDataFrame=True)``

    With the same seed, iterating with shuffle=False and num_workers=0 yields exactly the segments of
    ``np.random.seed(seed); segment_sequences(df, {"type": "random", ...}, AsDataFrame=True)``, in the same order.
    The segments are sampled once, in __init__, so every epoch uses the same segments (as the old pipeline did).

    Args:
        coverage: Expected number of times each base is covered; sets the number of sampled positions.
        seed: Seed of the legacy NumPy RNG (np.random.RandomState), the one np.random.seed() seeds.
        shuffle: Yield the segments in a random order instead of position order. Use it for training:
            position order puts segments of the same contig (and label) into the same batch. The order
            depends on seed and on the epoch set with set_epoch() (the HF Trainer calls it every epoch).
    """

    def __init__(
        self,
        *args,
        coverage: float = 1.0,
        seed: int | None = None,
        shuffle: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        if coverage <= 0:
            raise ValueError(f"Expected coverage > 0, got {coverage}.")
        self.coverage = coverage
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        # every DataLoader worker must shuffle with the same seed, otherwise their shares would overlap
        self._shuffle_seed = seed if seed is not None else int(np.random.SeedSequence().entropy % 2**32)

        self._sequence_ids, self._starts, self._ends = self._sample_segments()

    def _sample_segments(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sequence ids and absolute [start, end) coordinates of the kept segments, in position order."""
        total_len = self.sequence_dataset.get_sequence_len()
        n_segments = int(np.ceil(self.coverage * total_len / self.max_segment_length))

        # same RNG stream as np.random.seed(seed) followed by np.random.uniform(...)
        rng = np.random.RandomState(self.seed)
        positions = np.sort(np.int64(rng.uniform(0, total_len, n_segments)))

        contig_starts, contig_ends = self.sequence_dataset.get_contig_coordinates()
        # contig of each position: the first contig whose end is after the position
        sequence_ids = np.searchsorted(contig_ends, positions, side="right")
        # the segment starts at the position and stops at max_segment_length or at the contig end
        ends = np.minimum(positions + self.max_segment_length, contig_ends[sequence_ids])

        keep = ends - positions >= self.min_segment_length
        return sequence_ids[keep], positions[keep], ends[keep]

    def __len__(self) -> int:
        return len(self._sequence_ids)

    def set_epoch(self, epoch: int) -> None:
        """Called by the HF Trainer at the start of every epoch; changes the order when shuffle=True."""
        self.epoch = epoch

    def __iter__(self) -> Iterator[Segment | str]:
        worker = get_worker_info()
        worker_id, num_workers = (worker.id, worker.num_workers) if worker is not None else (0, 1)

        order = np.arange(len(self))
        if self.shuffle:
            np.random.RandomState(self._shuffle_seed + self.epoch).shuffle(order)

        # each worker takes every num_workers-th segment of the same order, so together they yield each segment once
        for k in order[worker_id::num_workers]:
            contig = self.sequence_dataset.get_contig_metadata_from_sequence_id(int(self._sequence_ids[k]))
            yield self.create_segment(int(k), contig, int(self._starts[k]), int(self._ends[k]))


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