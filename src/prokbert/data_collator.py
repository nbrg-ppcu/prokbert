from typing import Any

from dataclasses import dataclass

import torch

from prokbert.constants import SEQUENCE, SEGMENT_ID, SEQUENCE_ID, ABSOLUTE_COORDINATE
from prokbert.tokenizer import LCATokenizer


@dataclass
class SegmentDataCollator:
    """Turn a list of segments into a padded batch of model inputs.

    The segments must contain their sequence, i.e. the dataset has to be
    created with ``return_sequence=True``. By default only model inputs are
    returned, because the Hugging Face Trainer passes every key to
    ``model.forward``. Set ``return_metadata=True`` to also get segment and
    coordinate information.
    """

    tokenizer: LCATokenizer
    padding: bool = True
    return_tensors: str = "pt"
    return_metadata: bool = False
    add_special_tokens: bool = True


    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        if SEQUENCE not in features[0]:
            raise KeyError("Segments have no 'sequence'. Create the dataset with return_sequence=True.")

        batch = dict(
            self.tokenizer(
                [feature[SEQUENCE] for feature in features],
                padding = self.padding,
                return_tensors = self.return_tensors,
                add_special_tokens = self.add_special_tokens,
            )
        )
        if self.return_metadata:
            batch[SEGMENT_ID] = torch.tensor([feature[SEGMENT_ID] for feature in features])
            batch[SEQUENCE_ID] = torch.tensor([feature[SEQUENCE_ID] for feature in features])
            batch[ABSOLUTE_COORDINATE] = torch.tensor([feature[ABSOLUTE_COORDINATE] for feature in features])
        return batch





