import torch
from torch import nn


class CenterCost(nn.Module):
    """The center cost function used in the RTMDet paper.

    Used to quantify the closeness of grid points and a ground truth bbox.
    """

    def __init__(self, alpha: float = 10, beta: float = 3, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        """Initialize the cost class.

        alpha and beta are hyperparameters for the loss. See RTMDet paper for details:
        https://arxiv.org/abs/2212.07784
        """
        super().__init__(*args, **kwargs)

        self.alpha = alpha
        self.beta = beta

    def forward(self, gt_bbox: torch.Tensor, grid_points: torch.Tensor) -> torch.Tensor:
        """Calculate the cost between the gt_bbox and grid_points.

        ## Args
        - gt_bbox: tensor of shape (4) in format (x_min, y_min, x_max, y_max).
        - grid_points: tensor of shape (n, 2) in (x, y) format.
        ## Returns
        - a tensor of shape (n)
        """
        bbox_center = (gt_bbox[:2] + gt_bbox[2:]) / 2

        dist = (bbox_center - grid_points).pow(2).sum(dim=-1).sqrt()

        return torch.pow(self.alpha, dist - self.beta)
