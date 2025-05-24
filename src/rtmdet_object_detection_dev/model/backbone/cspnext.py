import torch
import torch.nn as nn

from ..basic_components import ConvModule, CSPLayer, SPPFBottleneck


class CSPNeXt(nn.Module):
    """
    Backbone of the RTMDet object detector.
    """

    def __init__(
        self, widen_factor: float, deepen_factor: float, *args, **kwargs
    ) -> None:
        """
        ## Args
        - widen_factor: the factor by which the out_channels should be multiplied.
        - deepen_factor: the factor by which the depth of the network should be multiplied.
        - All factors should be relative to the RTMDet large model.
        """
        super().__init__(*args, **kwargs)

        self.stem = nn.Sequential(
            ConvModule(
                in_channels=3,
                out_channels=round(32 * widen_factor),
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            ConvModule(
                in_channels=round(32 * widen_factor),
                out_channels=round(32 * widen_factor),
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            ConvModule(
                in_channels=round(32 * widen_factor),
                out_channels=round(64 * widen_factor),
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
        self.stage1 = nn.Sequential(
            ConvModule(
                in_channels=round(64 * widen_factor),
                out_channels=round(128 * widen_factor),
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            CSPLayer(
                in_channels=round(128 * widen_factor),
                out_channels=round(128 * widen_factor),
                add=True,
                n=round(3 * deepen_factor),
                attention=True,
            ),
        )
        self.stage2 = nn.Sequential(
            ConvModule(
                in_channels=round(128 * widen_factor),
                out_channels=round(256 * widen_factor),
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            CSPLayer(
                in_channels=round(256 * widen_factor),
                out_channels=round(256 * widen_factor),
                add=True,
                n=round(6 * deepen_factor),
                attention=True,
            ),
        )
        self.stage3 = nn.Sequential(
            ConvModule(
                in_channels=round(256 * widen_factor),
                out_channels=round(512 * widen_factor),
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            CSPLayer(
                in_channels=round(512 * widen_factor),
                out_channels=round(512 * widen_factor),
                add=True,
                n=round(6 * deepen_factor),
                attention=True,
            ),
        )
        self.stage4 = nn.Sequential(
            ConvModule(
                in_channels=round(512 * widen_factor),
                out_channels=round(1024 * widen_factor),
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            SPPFBottleneck(in_channels=round(1024 * widen_factor)),
            CSPLayer(
                in_channels=round(1024 * widen_factor),
                out_channels=round(1024 * widen_factor),
                add=False,
                n=round(3 * deepen_factor),
                attention=True,
            ),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.stage1(x)
        out1 = self.stage2(x)
        out2 = self.stage3(out1)
        out3 = self.stage4(out2)

        return out1, out2, out3
