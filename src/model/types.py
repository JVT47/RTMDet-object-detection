from dataclasses import dataclass
import torch

@dataclass
class RTMDetOutput:
    """
    A container class to hold the RTMDet model outputs. Note that in this project
    this class is used to hold predictions that are batched, i.e., of shape (B, C, H, W)
    and unbatched, i.e., of shape (C, H, W).
    """
    cls_preds: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    reg_preds: tuple[torch.Tensor, torch.Tensor, torch.Tensor]