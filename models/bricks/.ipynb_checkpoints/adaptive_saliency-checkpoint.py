import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    """
    Spatial Attention
    """

    def __init__(self, kernel_size=7):
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg = torch.mean(x, dim=1, keepdim=True)

        mx, _ = torch.max(x, dim=1, keepdim=True)

        attention = torch.cat([avg, mx], dim=1)

        attention = self.conv(attention)

        return self.sigmoid(attention)


class ChannelAttention(nn.Module):
    """
    Channel Attention
    """

    def __init__(self, channels, reduction=16):
        super().__init__()

        hidden = max(channels // reduction, 8)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.mlp = nn.Sequential(

            nn.Conv2d(channels, hidden, 1, bias=False),

            nn.ReLU(inplace=True),

            nn.Conv2d(hidden, channels, 1, bias=False)

        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        attention = self.avg_pool(x)

        attention = self.mlp(attention)

        return self.sigmoid(attention)


class AdaptiveFusion(nn.Module):
    """
    Learnable Softmax Fusion
    """

    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(2))

    def forward(self, spatial, channel):

        w = torch.softmax(self.weight, dim=0)

        attention = (

            w[0] * spatial

            +

            w[1] * channel

        )

        return attention


class AdaptiveSaliencyBlock(nn.Module):
    """
    Adaptive Saliency Enhancement
    """

    def __init__(self, channels):

        super().__init__()

        self.spatial = SpatialAttention()

        self.channel = ChannelAttention(channels)

        self.fusion = AdaptiveFusion()

    def forward(self, x):

        spatial = self.spatial(x)

        channel = self.channel(x)

        attention = self.fusion(spatial, channel)

        out = x * attention

        out = out + x

        return out