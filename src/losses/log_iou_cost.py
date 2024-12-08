import torch
import torch.nn as nn


class LogIoUCost(nn.Module):
    """
    The regression cost function used in RTMDet paper.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, IoU: torch.Tensor) -> torch.Tensor:
        """
        IoU: tensor of shape (n). Should contain the IoU score between the target bbox and bbox predictions.
        """
        loss = -torch.log(IoU)
        
        return loss