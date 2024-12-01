from dataclasses import dataclass
import torch

from src.processors.utils import BBoxLabelContainer, make_RTMDet_grid_points, transform_reg_pred_to_bbox_pred

@dataclass
class RTMDetOutput:
    """
    A container class to hold the RTMDet model outputs. Note that in this project
    this class is used to hold predictions that are batched, i.e., of shape (B, C, H, W)
    """
    cls_preds: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    reg_preds: tuple[torch.Tensor, torch.Tensor, torch.Tensor]

    def process_and_combine_layers(self) -> BBoxLabelContainer:
        """
        Combines the prediction from each layer in to one tensor and 
        flattens them to shapes (B, n, num_classes) and (B, n, 4), respectively. 
        Additonally, applies the sigmoid function to class predictions and transforms
        reg_preds to bboxes of form (x_min, y_min, x_max, y_max).
        """
        batch_size, num_classes, first_layer_height, first_layer_width = self.cls_preds[0].shape
        grid_points = make_RTMDet_grid_points(first_layer_height, first_layer_width)

        cls_preds = torch.cat([self.cls_preds[i].permute(0, 2, 3, 1).reshape(batch_size, -1, num_classes) for i in range(len(self.cls_preds))], dim=1)
        cls_preds = cls_preds.sigmoid()
        bbox_preds = torch.cat([self.reg_preds[i].permute(0, 2, 3, 1).reshape(batch_size, -1, 4) for i in range(len(self.reg_preds))], dim=1)
        bbox_preds = transform_reg_pred_to_bbox_pred(bbox_preds, grid_points)

        return BBoxLabelContainer(bbox_preds, cls_preds)