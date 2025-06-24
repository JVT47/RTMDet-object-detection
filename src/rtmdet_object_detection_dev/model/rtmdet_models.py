import torch
from torch import nn

from rtmdet_object_detection_dev.dataclasses.rtmdet_output import RTMDetOutput

from .backbone.cspnext import CSPNeXt
from .head.rtmdet_sep_bn_head import RTMDetSepBNHead
from .neck.cspnext_pafpn import CSPNeXtPAFPN


class RTMDet(nn.Module):
    """General RTMDet model."""

    def __init__(
        self,
        widen_factor: float,
        deepen_factor: float,
        num_classes: int,
        *args,  # noqa: ANN002
        exp_on_reg: bool,
        raw_output: bool = False,
        **kwargs,  # noqa: ANN003
    ) -> None:
        """Initialize the model."""
        super().__init__(*args, **kwargs)

        self.backbone = CSPNeXt(widen_factor, deepen_factor)
        self.neck = CSPNeXtPAFPN(widen_factor, deepen_factor)
        self.bbox_head = RTMDetSepBNHead(widen_factor, num_classes, exp_on_reg=exp_on_reg, raw_output=raw_output)

    def forward(
        self,
        x: torch.Tensor,
    ) -> (
        RTMDetOutput
        | tuple[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ]
    ):
        """Detect bboxes for the image(s)."""
        out = self.backbone(x)
        out = self.neck(out)

        return self.bbox_head(out)
