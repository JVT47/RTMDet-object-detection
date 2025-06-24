from dataclasses import dataclass

import torch


@dataclass
class BBoxLabelContainer:
    """A class that holds the bbox and label tensors."""

    bboxes: torch.Tensor
    labels: torch.Tensor
