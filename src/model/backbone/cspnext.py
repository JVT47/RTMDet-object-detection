import torch
import torch.nn as nn

from ..basic_components import ConvModule, CSPLayer, SPPFBottleneck


class CSPNeXt(nn.Module):
    """
    Backbone of the RTMDet object detector.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.stem = nn.Sequential(
            ConvModule(in_channels=3, out_channels=12, kernel_size=3, stride=2, padding=1),
            ConvModule(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1),
            ConvModule(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1),
        )
        self.stage1 = nn.Sequential(
            ConvModule(in_channels=24, out_channels=48, kernel_size=3, stride=2, padding=1),
            CSPLayer(in_channels=48, out_channels=48, add=True, n=1, attention=True),
        )
        self.stage2 = nn.Sequential(
            ConvModule(in_channels=48, out_channels=96, kernel_size=3, stride=2, padding=1),
            CSPLayer(in_channels=96, out_channels=96, add=True, n=1, attention=True),
        )
        self.stage3 = nn.Sequential(
            ConvModule(in_channels=96, out_channels=192, kernel_size=3, stride=2, padding=1),
            CSPLayer(in_channels=192, out_channels=192, add=True, n=1, attention=True),
        )
        self.stage4 = nn.Sequential(
            ConvModule(in_channels=192, out_channels=384, kernel_size=3, stride=2, padding=1),
            SPPFBottleneck(in_channels=384),
            CSPLayer(in_channels=384, out_channels=384, add=False, n=1, attention=True),
        )
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.stage1(x)
        out1 = self.stage2(x)
        out2 = self.stage3(out1)
        out3 = self.stage4(out2)

        return out1, out2, out3