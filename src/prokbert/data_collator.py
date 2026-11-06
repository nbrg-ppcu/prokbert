from typing import Any

from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence

from prokbert.tokenizer import LCATokenizer
from prokbert.constants import SEQUENCE, SEGMENT_ID, SEQUENCE_ID, ABSOLUTE_COORDINATE


from typing import Any, ClassVar


@dataclass
class DataCollator:
    """Turn a list of items (segments, genomes, ...) into a padded batch of model inputs.

    The items must contain their sequence (dataset created with ``return_sequence=True``) or,
    with ``has_embedding=True``, their embedding (``return_embedding=True``). By default only
    model inputs are returned, because the Hugging Face Trainer passes every key to
    ``model.forward``. Set ``return_metadata=True`` to also get the item's ``metadata_keys``.
    """
    tokenizer: LCATokenizer
    padding: bool = True
    return_tensors: str = "pt"
    has_embedding: bool = False
    return_metadata: bool = False
    add_special_tokens: bool = True

    metadata_keys: ClassVar[tuple[str, ...]] = ()

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        if self.has_embedding and "embedding" not in features[0]:
            raise KeyError("Items have no 'embedding'. Create the dataset with return_embedding=True.")
        if not self.has_embedding and SEQUENCE not in features[0]:
            raise KeyError("Items have no 'sequence'. Create the dataset with return_sequence=True.")

        if self.has_embedding:
            batch = {}
            batch["inputs_embeds"], batch["attention_mask"] = pad_embeddings(
                [feature["embedding"] for feature in features]
            )
        else:
            batch = dict(
                self.tokenizer(
                    [feature[SEQUENCE] for feature in features],
                    padding=self.padding,
                    return_tensors=self.return_tensors,
                    add_special_tokens=self.add_special_tokens,
                )
            )

        if "label" in features[0] and features[0]["label"] is not None:
            batch["labels"] = torch.tensor([feature["label"] for feature in features])

        if self.return_metadata:
            batch.update(self.collate_metadata(features))
        return batch

    def collate_metadata(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        """Metadata of the items as plain lists; override to return e.g. tensors."""
        return {key: [feature[key] for feature in features] for key in self.metadata_keys}


class SegmentDataCollator(DataCollator):
    """DataCollator for segments: the metadata (ids, coordinates) are numbers, returned as tensors."""
    metadata_keys: ClassVar[tuple[str, ...]] = (SEGMENT_ID, SEQUENCE_ID, ABSOLUTE_COORDINATE)

    def collate_metadata(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        return {key: torch.tensor([feature[key] for feature in features]) for key in self.metadata_keys}


class GenomeDataCollator(DataCollator):
    """DataCollator for genomes: the metadata (genome id, contig ids) are strings, returned as lists."""
    metadata_keys: ClassVar[tuple[str, ...]] = ("genome_id", "contig_ids")


def pad_embeddings(embeddings: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad segment embeddings with different numbers of rows to the longest one in the batch.

    Segments can overlap a different number of pooled rows, depending on their
    length and on where they start relative to the row boundaries, so their
    embeddings have different first dimensions.

    Args:
        embeddings: List of tensors, each of shape (n_rows_i, dim).

    Returns:
        padded: Tensor of shape (batch, max_rows, dim), zero-padded.
        mask: Tensor of shape (batch, max_rows), 1 for real rows and 0 for padding.
    """
    embeddings = [torch.as_tensor(e) for e in embeddings]
    lengths = torch.tensor([e.shape[0] for e in embeddings]) # number of rows per segment

    # a szegmensek beágyazásainak eltérő hossza lehet,
    # ezért a szegmensekhez hozzáadunk annyi nullvektort, amennyi szükséges
    # (mivel az nn.Embedding-be a padding_idx alapértelmezetten nulla, és ahhoz
    # egy nullvektor tartozik (amit nem tanítunk), ezért elegendő szintén nullvektort készíteni)
    padded = pad_sequence(embeddings, batch_first=True, padding_value=0.0)
    max_rows = padded.shape[1]
    # így a  további feldolgozás szükségessé válik a beágyazásokhoz is külön maszkot készíteni
    mask = (torch.arange(max_rows)[None, :] < lengths[:, None]).long()
    return padded, mask

