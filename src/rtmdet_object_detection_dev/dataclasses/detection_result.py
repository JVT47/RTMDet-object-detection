from dataclasses import dataclass

import torch


@dataclass
class DetectionResult:
    """A class to hold the bounding boxes, classes and confidence scores for an image."""

    bboxes: torch.Tensor  # (N, 4) N = number of positive detections
    classes: torch.Tensor  # (N)
    scores: torch.Tensor  # (N)
