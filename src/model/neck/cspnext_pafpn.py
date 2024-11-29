import torch
import torch.nn as nn

from ..basic_components import ConvModule, CSPLayer


class CSPNeXtPAFPN(nn.Module):
    """
    Neck of the RTMDet object detector.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.upsample = nn.Upsample(scale_factor=2.0, mode="nearest")
        self.reduce_layers = nn.ModuleList([
            ConvModule(in_channels=384, out_channels=192, kernel_size=1, stride=1, padding=0),
            ConvModule(in_channels=192, out_channels=96, kernel_size=1, stride=1, padding=0),
        ])
        self.top_down_blocks = nn.ModuleList([
            CSPLayer(in_channels=384, out_channels=192, add=False, n=1, attention=False),
            CSPLayer(in_channels=192, out_channels=96, add=False, n=1, attention=False),
        ])
        self.downsamples = nn.ModuleList([
            ConvModule(in_channels=96, out_channels=96, kernel_size=3, stride=2, padding=1),
            ConvModule(in_channels=192, out_channels=192, kernel_size=3, stride=2, padding=1),
        ])
        self.bottom_up_blocks = nn.ModuleList([
            CSPLayer(in_channels=192, out_channels=192, add=False, n=1, attention=False),
            CSPLayer(in_channels=384, out_channels=384, add=False, n=1, attention=False),
        ])
        self.out_convs = nn.ModuleList([
            ConvModule(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
            ConvModule(in_channels=192, out_channels=96, kernel_size=3, stride=1, padding=1),
            ConvModule(in_channels=384, out_channels=96, kernel_size=3, stride=1, padding=1),
        ])
    
    def forward(self, x: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # top down path
        out3 = self.reduce_layers[0](x[2])

        out2 = self.upsample(out3)
        out2 = torch.concat([out2, x[1]], dim=1)
        out2 = self.top_down_blocks[0](out2)
        out2 = self.reduce_layers[1](out2)

        out1 = self.upsample(out2)
        out1 = torch.concat([out1, x[0]], dim=1)
        out1 = self.top_down_blocks[1](out1)

        # bottom up path
        tmp1 = self.downsamples[0](out1)
        out2 = torch.concat([tmp1, out2], dim=1)
        out2 = self.bottom_up_blocks[0](out2)
        tmp2 = self.downsamples[1](out2)

        out3 = torch.concat([tmp2, out3], dim=1)
        out3 = self.bottom_up_blocks[1](out3)

        # outputs
        out1 = self.out_convs[0](out1)
        out2 = self.out_convs[1](out2)
        out3 = self.out_convs[2](out3)

        return out1, out2, out3