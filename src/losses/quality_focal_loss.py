import torch
import torch.nn as nn


class QualityFocalLoss(nn.Module):
    """
    Implementation of QualityFocalLoss (QFL).
    Based on https://arxiv.org/abs/2006.04388
    """
    def __init__(self, beta: float = 2, *args, **kwargs) -> None:
        """
        beta: a scaling factor which in RTMDet is 2.
        """
        super().__init__(*args, **kwargs)

        self.beta = beta
    
    def forward(self, pred_label: torch.Tensor, target_label: torch.Tensor, IoU: torch.Tensor) -> torch.Tensor:
        """
        pred_label: tensor of shape (B, n, c) or (n, c) where n is the number of predictions and c the number of classes.
        target_label: tensor of shape (B, n, c) or (n, c). Usually the targets are one-hot encoded.
        IoU: tensorf of shape (B, n) or (n). The IoU score between the prediction and target. Used to calculate the soft label.
        """
        soft_label = target_label * IoU.unsqueeze(-1)
        
        loss = -1 * torch.abs(pred_label - soft_label).pow(self.beta)
        loss *= (1 - soft_label)*torch.log(1 - pred_label) + soft_label * torch.log(pred_label)

        loss = loss.sum(dim=-1)

        return loss