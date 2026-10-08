"""
Segment samplers.

Conceptually, *sampling* (deciding which regions of the genome become
segments) and *loading* (producing those segments for training) are two
separate steps.

Here the two steps are deliberately combined: ``RandomSegmentSampler`` and
``ContiguousSegmentSampler`` are aliases of
:class:`prokbert.dataset.RandomSegmentDataset` and
:class:`prokbert.dataset.ContiguousSegmentDataset`, which are
``IterableDataset``s that both choose the segments and yield them.

This keeps the implementation simple:

- One class per segmentation strategy, with no extra index type and no
  separate dataset class to keep in sync.
- Segments are produced inside the DataLoader workers, so sampling runs in
  parallel instead of in the main process.
- Worker handling is minimal: random sampling needs no coordination (each
  worker draws independently), and contiguous sampling only splits the contigs
  between workers with a single slice.

All logic and API documentation live in ``prokbert.dataset``; these names exist
so the sampling step can be referred to as a sampler.

Usage notes
-----------
- These are not ``torch.utils.data.Sampler`` subclasses. You can pass them to
  ``DataLoader`` as the dataset, not as ``sampler=``.
- ``RandomSegmentSampler`` is an infinite stream and has no length. Limit it
  with ``itertools.islice`` or a step count (e.g. ``max_steps``).
- ``ContiguousSegmentSampler`` does not shuffle (currently).
- ``segment_id``: deterministic for contiguous sampling (the same segment
  always gets the same ID). For random sampling it numbers the draws, not the
  regions: it restarts from 0 on every new iteration and is unique across
  workers within one iteration.
- Coordinates are 0-based and half-open, ``[start, end)``.
- With ``return_sequence=True`` each segment also contains its sequence under
  the ``"sequence"`` key.
"""

from prokbert.dataset import ContiguousSegmentDataset, RandomSegmentDataset

__all__ = ["RandomSegmentSampler", "ContiguousSegmentSampler"]

RandomSegmentSampler = RandomSegmentDataset
ContiguousSegmentSampler = ContiguousSegmentDataset