from typing import Any

from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence

from prokbert.tokenizer import LCATokenizer
from prokbert.constants import SEQUENCE, SEGMENT_ID, SEQUENCE_ID, ABSOLUTE_COORDINATE


from typing import Any, ClassVar


@dataclass
class DataCollator:
    """Turn a list of items (e.g. segments or genomes) into a padded batch of model inputs.

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
    pooling: str | None = None    # "mean", "sum": pool the embedding rows again, before padding the batch
    pooling_len: int = 1          # number of consecutive rows pooled into one

    metadata_keys: ClassVar[tuple[str, ...]] = ()

    def __post_init__(self) -> None:
        if self.pooling not in (None, "mean", "sum", "full"):
            raise ValueError(f"pooling must be None, 'mean' or 'sum', got {self.pooling!r}")
        if self.pooling is not None and self.pooling_len < 1:
            raise ValueError(f"pooling_len must be at least 1, got {self.pooling_len}")

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        if self.has_embedding and "embedding" not in features[0]:
            raise KeyError("Items have no 'embedding'. Create the dataset with return_embedding=True.")
        if not self.has_embedding and SEQUENCE not in features[0]:
            raise KeyError("Items have no 'sequence'. Create the dataset with return_sequence=True.")

        if self.has_embedding:
            embeddings = [torch.as_tensor(feature["embedding"]) for feature in features]
            if self.pooling is not None:
                embeddings = [pool_rows(e, self.pooling_len, self.pooling) for e in embeddings]

            batch = {}
            batch["inputs_embeds"], batch["attention_mask"] = pad_embeddings(embeddings)
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

    # a beágyazások eltérő számú sorból állhatnak, ezért a rövidebbeket nullvektorokkal
    # egészítjük ki a batch leghosszabbjáig; a nullák értéke nem számít, mert a maszk
    # (0 = padding) alapján a további feldolgozás figyelmen kívül hagyja őket
    padded = pad_sequence(embeddings, batch_first=True, padding_value=0.0)
    max_rows = padded.shape[1]
    # így a  további feldolgozás szükségessé válik a beágyazásokhoz is külön maszkot készíteni
    mask = (torch.arange(max_rows)[None, :] < lengths[:, None]).long()
    return padded, mask

def pool_rows(embedding: torch.Tensor, pooling_len: int, pooling: str) -> torch.Tensor:
    """
    Pool every pooling_len consecutive rows of one item: (n_rows, dim) -> (ceil(n_rows / pooling_len), dim).

    The last block may have fewer than pooling_len rows; it is pooled over its own rows, without padding.
    E.g. with pooling_len=5: (100, dim) -> (20, dim), (101, dim) -> (21, dim) where the last row pools 1 row.
    """
    n_rows, dim = embedding.shape
    n_full = (n_rows // pooling_len) * pooling_len       # rows that fill complete blocks
    parts = []

    if n_full > 0:
        full = embedding[:n_full].reshape(-1, pooling_len, dim)                      # (n_full_blocks, pooling_len, dim)
        parts.append(full.mean(dim=1) if pooling == "mean" else full.sum(dim=1))     # (n_full_blocks, dim)

    if n_full < n_rows:
        rest = embedding[n_full:]                                                     # (n_rest_rows, dim), < pooling_len rows
        parts.append((rest.mean(dim=0) if pooling == "mean" else rest.sum(dim=0)).unsqueeze(0))   # (1, dim)

    return torch.cat(parts, dim=0) if parts else embedding.new_zeros(0, dim)

