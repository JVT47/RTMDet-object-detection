from dataclasses import dataclass
import torch

from src.processors.utils import BBoxLabelContainer

@dataclass
class RTMDetOutput:
    """
    A container class to hold the RTMDet model outputs. Note that in this project
    this class is used to hold predictions that are batched, i.e., of shape (B, C, H, W)
    """
    cls_preds: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    reg_preds: tuple[torch.Tensor, torch.Tensor, torch.Tensor]

    def flatten(self) -> BBoxLabelContainer:
        """
        Combines the prediction from each layer in to one tensor and 
        flattens them to shapes (B, n, num_classes) and (B, n, 4), respectively.
        """
        batch_size, num_classes, _, _ = self.cls_preds[0].shape
        cls_preds = torch.cat([self.cls_preds[i].permute(0, 2, 3, 1).reshape(batch_size, -1, num_classes) for i in range(len(self.cls_preds))], dim=1)
        reg_preds = torch.cat([self.reg_preds[i].permute(0, 2, 3, 1).reshape(batch_size, -1, 4) for i in range(len(self.reg_preds))], dim=1)

        return BBoxLabelContainer(reg_preds, cls_preds)