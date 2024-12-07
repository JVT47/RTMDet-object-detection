import torch
import torch.nn as nn

from ..basic_components import ConvModule
from src.dataclasses.rtmdet_output import RTMDetOutput


class RTMDetSepBNHead(nn.Module):
    """
    Bounding box head of the RTMDet detector.
    """
    def __init__(self, widen_factor: float, num_classes: int, *args, **kwargs) -> None:
        """
        widen_factor: the factor by which the out_channels should be multiplied.
        All factor should be relative to RTMDet large model. 
        num_classes: the number of classes the model should predict.
        """
        super().__init__(*args, **kwargs)

        self.cls_convs = nn.ModuleList([
            *[nn.Sequential(
                ConvModule(in_channels=round(256 * widen_factor), out_channels=round(256 * widen_factor), kernel_size=3, stride=1, padding=1),
                ConvModule(in_channels=round(256 * widen_factor), out_channels=round(256 * widen_factor), kernel_size=3, stride=1, padding=1)
            ) for _ in range(3)]
        ])
        self.reg_convs = nn.ModuleList([
            *[nn.Sequential(
                ConvModule(in_channels=round(256 * widen_factor), out_channels=round(256 * widen_factor), kernel_size=3, stride=1, padding=1),
                ConvModule(in_channels=round(256 * widen_factor), out_channels=round(256 * widen_factor), kernel_size=3, stride=1, padding=1),
            ) for _ in range(3)]
        ])
        self.rtm_cls = nn.ModuleList([
            *[nn.Conv2d(in_channels=round(256 * widen_factor), out_channels=num_classes, kernel_size=1, stride=1) for _ in range(3)]
        ])
        self.rtm_reg = nn.ModuleList([
            *[nn.Conv2d(in_channels=round(256 * widen_factor), out_channels=4, kernel_size=1, stride=1) for _ in range(3)]
        ])
        # share conv weights 
        for i in range(3):
            for j in range(2):
                self.cls_convs[i][j].conv = self.cls_convs[0][j].conv # type: ignore In this case ModuleList contains Sequential so double indexing is ok 
                self.reg_convs[i][j].conv = self.reg_convs[0][j].conv # type: ignore
        
    def forward(self, x: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> RTMDetOutput:
        cls_scores = []
        reg_preds = []
        for i, out_scale in enumerate([8, 16, 32]): # Output scales for bbox regression
            cls_score = self.cls_convs[i](x[i])
            cls_score = self.rtm_cls[i](cls_score)

            reg_pred = self.reg_convs[i](x[i])
            reg_pred = self.rtm_reg[i](reg_pred)

            cls_scores.append(cls_score)
            reg_preds.append(reg_pred * out_scale)

        return RTMDetOutput(tuple(cls_scores), tuple(reg_preds))