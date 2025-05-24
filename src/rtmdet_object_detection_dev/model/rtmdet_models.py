import torch
import torch.nn as nn

from .backbone.cspnext import CSPNeXt
from .neck.cspnext_pafpn import CSPNeXtPAFPN
from .head.rtmdet_sep_bn_head import RTMDetSepBNHead
from rtmdet_object_detection_dev.dataclasses.rtmdet_output import RTMDetOutput


class RTMDet(nn.Module):
    """
    General RTMDet model.
    """

    def __init__(
        self,
        widen_factor: float,
        deepen_factor: float,
        num_classes: int,
        exp_on_reg: bool,
        *args,
        raw_output: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.backbone = CSPNeXt(widen_factor, deepen_factor)
        self.neck = CSPNeXtPAFPN(widen_factor, deepen_factor)
        self.bbox_head = RTMDetSepBNHead(
            widen_factor, num_classes, exp_on_reg=exp_on_reg, raw_output=raw_output
        )

    def forward(
        self, x: torch.Tensor
    ) -> (
        RTMDetOutput
        | tuple[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ]
    ):
        out = self.backbone(x)
        out = self.neck(out)
        out = self.bbox_head(out)

        return out
