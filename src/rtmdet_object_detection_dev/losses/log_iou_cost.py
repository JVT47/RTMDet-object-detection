import torch
from torch import nn


class LogIoUCost(nn.Module):
    """The regression cost function used in the RTMDet paper."""

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        """Initialize the cost."""
        super().__init__(*args, **kwargs)

    def forward(self, IoU: torch.Tensor) -> torch.Tensor:  # noqa: N803
        """Calculate the loss for the given IoUs.

        ## Args
        - IoU: tensor of shape (n). Should contain the IoU score between the target bbox and bbox predictions.
        """
        return -torch.log(IoU)
