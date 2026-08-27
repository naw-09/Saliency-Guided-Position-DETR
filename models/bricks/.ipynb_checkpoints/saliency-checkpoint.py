import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    """
    Spatial attention module.

    Generates a spatial attention map with shape:
        [B, 1, H, W]

    It uses channel-wise average pooling and max pooling,
    followed by a convolution and sigmoid activation.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()

        if kernel_size not in (3, 7):
            raise ValueError(
                f"kernel_size must be 3 or 7, but received {kernel_size}"
            )

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input feature tensor with shape [B, C, H, W]

        Returns:
            Spatial attention map with shape [B, 1, H, W]
        """

        avg_feature = torch.mean(
            x,
            dim=1,
            keepdim=True,
        )

        max_feature, _ = torch.max(
            x,
            dim=1,
            keepdim=True,
        )

        pooled_feature = torch.cat(
            [avg_feature, max_feature],
            dim=1,
        )

        attention = self.conv(pooled_feature)

        return self.sigmoid(attention)


class ChannelAttention(nn.Module):
    """
    Channel attention module.

    Generates a channel attention map with shape:
        [B, C, 1, 1]

    It uses global average pooling followed by a small MLP
    implemented with 1x1 convolutions.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
    ):
        super().__init__()

        if channels <= 0:
            raise ValueError(
                f"channels must be positive, but received {channels}"
            )

        if reduction <= 0:
            raise ValueError(
                f"reduction must be positive, but received {reduction}"
            )

        hidden_channels = max(
            channels // reduction,
            8,
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=hidden_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=channels,
                kernel_size=1,
                bias=False,
            ),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input feature tensor with shape [B, C, H, W]

        Returns:
            Channel attention map with shape [B, C, 1, 1]
        """

        attention = self.avg_pool(x)
        attention = self.mlp(attention)

        return self.sigmoid(attention)


class AdaptiveFusion(nn.Module):
    """
    Learnable fusion of spatial and channel attention.

    Two trainable scalar weights are normalized using softmax.

    Spatial attention:
        [B, 1, H, W]

    Channel attention:
        [B, C, 1, 1]

    PyTorch broadcasting produces:
        [B, C, H, W]
    """

    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(
            torch.ones(2, dtype=torch.float32)
        )

    def forward(
        self,
        spatial_attention: torch.Tensor,
        channel_attention: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            spatial_attention: Tensor with shape [B, 1, H, W]
            channel_attention: Tensor with shape [B, C, 1, 1]

        Returns:
            Fused attention tensor with shape [B, C, H, W]
        """

        fusion_weight = torch.softmax(
            self.weight,
            dim=0,
        )

        fused_attention = (
            fusion_weight[0] * spatial_attention
            + fusion_weight[1] * channel_attention
        )

        return fused_attention


class AdaptiveSaliencyBlock(nn.Module):
    """
    Feature-level adaptive saliency enhancement block.

    The block combines:
        1. Spatial attention
        2. Channel attention
        3. Learnable softmax fusion
        4. Learnable residual scaling

    The learnable gamma parameter is initialized to zero.
    Therefore, the block starts as an identity mapping:

        output = input

    During training, gamma can learn how strongly the
    saliency-enhanced feature should be added.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        spatial_kernel_size: int = 3,
    ):
        super().__init__()

        self.spatial = SpatialAttention(
            kernel_size=spatial_kernel_size
        )

        self.channel = ChannelAttention(
            channels=channels,
            reduction=reduction,
        )

        self.fusion = AdaptiveFusion()

        # Start from identity mapping for stable fine-tuning.
        self.gamma = nn.Parameter(
            torch.zeros(1, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input feature tensor with shape [B, C, H, W]

        Returns:
            Enhanced feature tensor with the same shape as x
        """

        if x.ndim != 4:
            raise ValueError(
                "AdaptiveSaliencyBlock expects a 4D tensor "
                f"[B, C, H, W], but received shape {tuple(x.shape)}"
            )

        spatial_attention = self.spatial(x)

        channel_attention = self.channel(x)

        fused_attention = self.fusion(
            spatial_attention,
            channel_attention,
        )

        enhanced_feature = (
            x
            + self.gamma * x * fused_attention
        )

        return enhanced_feature



