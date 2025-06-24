import torch
from torch import nn


class GIoULoss(nn.Module):
    """Implementation of the generalized intersection over union loss."""

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        """Initialize the loss."""
        super().__init__(*args, **kwargs)

    def forward(self, boxes_1: torch.Tensor, boxes_2: torch.Tensor) -> torch.Tensor:
        """Calculate the loss between the given bounding boxes.

        Both inputs should be in shape (B, n, 4). Returns a tensor of shape (B, n).
        """
        IoU, union = self.calc_IoU_and_union(boxes_1, boxes_2)  # noqa: N806

        enclosure_width = torch.maximum(boxes_1[..., -2], boxes_2[..., -2]) - torch.minimum(
            boxes_1[..., -4],
            boxes_2[..., -4],
        )
        enclosure_height = torch.maximum(boxes_1[..., -1], boxes_2[..., -1]) - torch.minimum(
            boxes_1[..., -3],
            boxes_2[..., -3],
        )
        enclosure_area = enclosure_width * enclosure_height

        GIoU = IoU - (enclosure_area - union) / enclosure_area  # noqa: N806
        return 1 - GIoU

    def calc_IoU_and_union(self, boxes_1: torch.Tensor, boxes_2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: N802
        """Calculate the IoU and union between two sets of bounding boxes so that the GIoU can be calculated.

        Both inputs should be in shape (..., 4) and the calculations are done element wise.
        Returns tensors of shape (...), i.e., with one dimension less than the input. (IoU, Union).
        """
        I_width = torch.minimum(boxes_1[..., -2], boxes_2[..., -2]) - torch.maximum(boxes_1[..., -4], boxes_2[..., -4])  # noqa: N806
        I_width = torch.maximum(torch.tensor([0]), I_width)  # noqa: N806
        I_height = torch.minimum(boxes_1[..., -1], boxes_2[..., -1]) - torch.maximum(boxes_1[..., -3], boxes_2[..., -3])  # noqa: N806
        I_height = torch.maximum(torch.tensor([0]), I_height)  # noqa: N806
        I_area = I_width * I_height  # noqa: N806

        boxes_1_area = (boxes_1[..., -2] - boxes_1[..., -4]) * (boxes_1[..., -1] - boxes_1[..., -3])
        boxes_1_area = torch.maximum(torch.tensor([0]), boxes_1_area)
        boxes_2_area = (boxes_2[..., -2] - boxes_2[..., -4]) * (boxes_2[..., -1] - boxes_2[..., -3])
        boxes_2_area = torch.maximum(torch.tensor([0]), boxes_2_area)

        union = boxes_1_area + boxes_2_area - I_area
        IoU = torch.zeros_like(union)  # noqa: N806
        mask = union > 0
        IoU[mask] = I_area[mask] / union[mask]

        return IoU, union
