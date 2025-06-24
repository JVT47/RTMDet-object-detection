from dataclasses import dataclass

import torch

from .bbox_label_container import BBoxLabelContainer


@dataclass
class RTMDetOutput:
    """A container class to hold the RTMDet model outputs.

    Note that in this project this class is used to hold predictions that are batched, i.e., of shape (B, C, H, W).
    """

    cls_preds: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    reg_preds: tuple[torch.Tensor, torch.Tensor, torch.Tensor]

    def create_model_grid_points(self) -> torch.Tensor:
        """Create a tensor with the grid point locations corresponding to the models output.

        Returns a tensor of shape (n, 2). Where n is the total number of grid points. Note that
        some of the grid points overlap since the model predict in multiple scales.
        """
        scales = [8, 16, 32]
        grid_points = []
        _, _, height, width = self.cls_preds[0].shape

        for scale in scales:
            points_xx, points_yy = torch.meshgrid(
                torch.arange(0, width, dtype=torch.float),
                torch.arange(0, height, dtype=torch.float),
                indexing="xy",
            )
            points = (
                torch.stack([points_xx.flatten(), points_yy.flatten()], dim=-1) * scale
            )  # shape (H * W, 2) in (x, y) format
            grid_points.append(points)

            height /= 2
            width /= 2

        return torch.cat(grid_points, dim=0)

    def process_and_combine_layers(self) -> BBoxLabelContainer:
        """Combine the prediction from each layer in to one tensor and flatten.

        Reshapes cls_preds and reg_preds to shapes (B, n, num_classes) and (B, n, 4), respectively.
        Additionally, applies the sigmoid function to class predictions and transforms
        reg_preds to bboxes of form (x_min, y_min, x_max, y_max).
        """
        batch_size, num_classes, _, _ = self.cls_preds[0].shape
        grid_points = self.create_model_grid_points()

        cls_preds = torch.cat(
            [
                self.cls_preds[i].permute(0, 2, 3, 1).reshape(batch_size, -1, num_classes)
                for i in range(len(self.cls_preds))
            ],
            dim=1,
        )
        cls_preds = cls_preds.sigmoid()
        bbox_preds = torch.cat(
            [self.reg_preds[i].permute(0, 2, 3, 1).reshape(batch_size, -1, 4) for i in range(len(self.reg_preds))],
            dim=1,
        )
        bbox_preds = self.transform_reg_pred_to_bbox_pred(bbox_preds, grid_points)

        return BBoxLabelContainer(bbox_preds, cls_preds)

    @staticmethod
    def transform_reg_pred_to_bbox_pred(reg_pred: torch.Tensor, grid_points: torch.Tensor) -> torch.Tensor:
        """Transform reg_pred from the model into bbox coordinates of the form (x_min, y_min, x_max, y_max).

        # Args
        - reg_pred: tensor of shape (B, n, 4)
        - grid_points: tensor of shape (n, 2)
        """
        batch_size, _, _ = reg_pred.shape
        bbox_pred = torch.cat([grid_points, grid_points], dim=-1)
        bbox_pred = bbox_pred.unsqueeze(0).repeat(batch_size, 1, 1)

        bbox_pred[..., :2] -= reg_pred[..., :2]
        bbox_pred[..., 2:] += reg_pred[..., 2:]

        return bbox_pred
