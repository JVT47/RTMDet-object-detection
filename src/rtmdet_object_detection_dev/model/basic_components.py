import torch
from torch import nn


class ConvModule(nn.Module):
    """Basic convolution module."""

    def __init__(  # noqa: PLR0913
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int | str,
        groups: int = 1,
        *args,  # noqa: ANN002
        **kwargs,  # noqa: ANN003
    ) -> None:
        """Initialize the module."""
        super().__init__(*args, **kwargs)

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.activate = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform convolution, batchnorm, and activation."""
        x = self.conv(x)
        x = self.bn(x)
        return self.activate(x)


class DepthwiseSeparableConvModule(nn.Module):
    """Depthwise separable convolution module."""

    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        stride: int,
        padding: int | str,
        *args,  # noqa: ANN002
        **kwargs,  # noqa: ANN003
    ) -> None:
        """Initialize the module."""
        super().__init__(*args, **kwargs)

        self.depthwise_conv = ConvModule(
            in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
        )
        self.pointwise_conv = ConvModule(in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform depthwise, and pointwise convolution."""
        x = self.depthwise_conv(x)

        return self.pointwise_conv(x)


class CSPNeXtBlock(nn.Module):
    """CSPNeXtBlock module."""

    def __init__(self, in_channels: int, out_channels: int, *args, add: bool, **kwargs) -> None:  # noqa: ANN002, ANN003
        """Initialize the module."""
        super().__init__(*args, **kwargs)

        self.conv1 = ConvModule(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = DepthwiseSeparableConvModule(in_channels=out_channels, kernel_size=5, stride=1, padding=2)
        self.add = add

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform the module calculations."""
        out = self.conv1(x)
        out = self.conv2(out)

        if self.add:
            out += x

        return out


class ChannelAttention(nn.Module):
    """Channel attention module."""

    def __init__(self, in_channels: int, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        """Initialize the module."""
        super().__init__(*args, **kwargs)

        self.global_avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Conv2d(in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.act = nn.Hardsigmoid(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform channel attention."""
        out = self.global_avgpool(x)
        out = self.fc(out)
        out = self.act(out)

        return torch.einsum("bchw,bcij -> bchw", x, out)


class CSPLayer(nn.Module):
    """CSPLayer module."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n: int,
        *args,  # noqa: ANN002
        add: bool,
        attention: bool,
        **kwargs,  # noqa: ANN003
    ) -> None:
        """Initialize the module."""
        super().__init__(*args, **kwargs)

        self.main_conv = ConvModule(in_channels, out_channels // 2, kernel_size=1, stride=1, padding=0)
        self.short_conv = ConvModule(in_channels, out_channels // 2, kernel_size=1, stride=1, padding=0)
        self.final_conv = ConvModule(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.blocks = nn.Sequential(
            *[
                CSPNeXtBlock(
                    in_channels=out_channels // 2,
                    out_channels=out_channels // 2,
                    add=add,
                )
                for _ in range(n)
            ],
        )
        self.attention = ChannelAttention(in_channels=out_channels) if attention else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform the module calculations."""
        main = self.main_conv(x)
        main = self.blocks(main)
        short = self.short_conv(x)

        out = torch.concat([main, short], dim=1)
        if self.attention is not None:
            out = self.attention(out)

        return self.final_conv(out)


class SPPFBottleneck(nn.Module):
    """SPPFBottleneck module."""

    def __init__(self, in_channels: int, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        """Initialize the module."""
        super().__init__(*args, **kwargs)

        self.conv1 = ConvModule(
            in_channels,
            out_channels=in_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.poolings = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False),
                nn.MaxPool2d(kernel_size=9, stride=1, padding=4, dilation=1, ceil_mode=False),
                nn.MaxPool2d(kernel_size=13, stride=1, padding=6, dilation=1, ceil_mode=False),
            ],
        )
        self.conv2 = ConvModule(
            (in_channels // 2) * 4,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform module calculations."""
        x = self.conv1(x)
        pooled = [pool(x) for pool in self.poolings]
        out = torch.cat([x, *pooled], dim=1)
        return self.conv2(out)
