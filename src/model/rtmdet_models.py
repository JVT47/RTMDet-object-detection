from enum import Enum
import torch
import torch.nn as nn

from .backbone.cspnext import CSPNeXt
from .neck.cspnext_pafpn import CSPNeXtPAFPN
from .head.rtmdet_sep_bn_head import RTMDetSepBNHead
from src.dataclasses.rtmdet_output import RTMDetOutput
    

class RTMDetTiny(nn.Module):
    """
    RTMDet tiny model.
    """
    def __init__(self, num_classes: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        widen_factor = 0.375
        deepen_factor = 0.167
        self.backbone = CSPNeXt(widen_factor, deepen_factor)
        self.neck = CSPNeXtPAFPN(widen_factor, deepen_factor)
        self.bbox_head = RTMDetSepBNHead(widen_factor, num_classes, exp_on_reg=False)

    def forward(self, x: torch.Tensor) -> RTMDetOutput:
        out = self.backbone(x)
        out = self.neck(out)

        return self.bbox_head(out)


class RTMDetS(nn.Module):
    """
    RTMDet small model.
    """
    def __init__(self, num_classes: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        widen_factor = 0.5
        deepen_factor = 0.33
        self.backbone = CSPNeXt(widen_factor, deepen_factor)
        self.neck = CSPNeXtPAFPN(widen_factor, deepen_factor)
        self.bbox_head = RTMDetSepBNHead(widen_factor, num_classes, exp_on_reg=False)

    def forward(self, x: torch.Tensor) -> RTMDetOutput:
        out = self.backbone(x)
        out = self.neck(out)

        return self.bbox_head(out)


class RTMDetM(nn.Module):
    """
    RTMDet medium model.
    """
    def __init__(self, num_classes: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        widen_factor = 0.75
        deepen_factor = 0.67
        self.backbone = CSPNeXt(widen_factor, deepen_factor)
        self.neck = CSPNeXtPAFPN(widen_factor, deepen_factor)
        self.bbox_head = RTMDetSepBNHead(widen_factor, num_classes, exp_on_reg=True)

    def forward(self, x: torch.Tensor) -> RTMDetOutput:
        out = self.backbone(x)
        out = self.neck(out)

        return self.bbox_head(out)


class RTMDetL(nn.Module):
    """
    RTMDet large model.
    """
    def __init__(self, num_classes: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        widen_factor = 1.0
        deepen_factor = 1.0
        self.backbone = CSPNeXt(widen_factor, deepen_factor)
        self.neck = CSPNeXtPAFPN(widen_factor, deepen_factor)
        self.bbox_head = RTMDetSepBNHead(widen_factor, num_classes, exp_on_reg=True)

    def forward(self, x: torch.Tensor) -> RTMDetOutput:
        out = self.backbone(x)
        out = self.neck(out)

        return self.bbox_head(out)


class RTMDetX(nn.Module):
    """
    RTMDet xl model.
    """
    def __init__(self, num_classes: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        widen_factor = 1.25
        deepen_factor = 1.33
        self.backbone = CSPNeXt(widen_factor, deepen_factor)
        self.neck = CSPNeXtPAFPN(widen_factor, deepen_factor)
        self.bbox_head = RTMDetSepBNHead(widen_factor, num_classes, exp_on_reg=True)

    def forward(self, x: torch.Tensor) -> RTMDetOutput:
        out = self.backbone(x)
        out = self.neck(out)

        return self.bbox_head(out)