import torch
import torch.nn as nn

from .backbone.cspnext import CSPNeXt
from .neck.cspnext_pafpn import CSPNeXtPAFPN
from .head.rtmdet_sep_bn_head import RTMDetSepBNHead
from .types import RTMDetOutput



class RTMDetTiny(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.backbone = CSPNeXt()
        self.neck = CSPNeXtPAFPN()
        self.bbox_head = RTMDetSepBNHead()
    
    def forward(self, x: torch.Tensor) -> RTMDetOutput:
        out: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = self.backbone(x)
        out = self.neck(out)

        return self.bbox_head(out)