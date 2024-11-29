import torch
import torch.nn as nn


class ConvModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int | str, groups: int = 1,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.activate = nn.SiLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activate(x)

        return x


class DepthwiseSeparableConvModule(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int, padding: int | str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.depthwise_conv = ConvModule(in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels)
        self.pointwise_conv = ConvModule(in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        return x


class CSPNeXtBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, add: bool, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.conv1 = ConvModule(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = DepthwiseSeparableConvModule(in_channels=out_channels, kernel_size=5, stride=1, padding=2)
        self.add = add
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)

        if self.add:
            out += x
        
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.global_avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Conv2d(in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.act = nn.Hardsigmoid(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.global_avgpool(x)
        out = self.fc(out)
        out = self.act(out)
        out = torch.einsum("bchw,bcij -> bchw", x, out)

        return out


class CSPLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, add: bool, n: int, attention: bool, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.main_conv = ConvModule(in_channels, out_channels // 2, kernel_size=1, stride=1, padding=0)
        self.short_conv = ConvModule(in_channels, out_channels // 2, kernel_size=1, stride=1, padding=0)
        self.final_conv = ConvModule(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.blocks = nn.Sequential(
            *[CSPNeXtBlock(in_channels=out_channels // 2, out_channels=out_channels // 2, add=add) for _ in range(n)]
        )
        self.attention = ChannelAttention(in_channels=out_channels) if attention else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main = self.main_conv(x)
        main = self.blocks(main)
        short = self.short_conv(x)

        out = torch.concat([main, short], dim=1)
        if self.attention is not None:
            out = self.attention(out)
        out = self.final_conv(out)

        return out


class SPPFBottleneck(nn.Module):
    def __init__(self, in_channels: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.conv1 = ConvModule(in_channels, out_channels=in_channels // 2, kernel_size=1, stride=1, padding=0)
        self.poolings = nn.ModuleList([
            nn.MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False),
            nn.MaxPool2d(kernel_size=9, stride=1, padding=4, dilation=1, ceil_mode=False),
            nn.MaxPool2d(kernel_size=13, stride=1, padding=6, dilation=1, ceil_mode=False),
        ])
        self.conv2 = ConvModule((in_channels // 2) * 4, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        pooling1 = self.poolings[0](x)
        pooling2 = self.poolings[1](pooling1)
        pooling3 = self.poolings[2](pooling2)

        out = torch.concat([x, pooling1, pooling2, pooling3], dim=1)
        out = self.conv2(out)

        return out