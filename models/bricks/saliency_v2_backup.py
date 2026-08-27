import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    """Generate spatial attention with shape [B, 1, H, W]."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()

        if kernel_size not in (3, 7):
            raise ValueError(
                f"kernel_size must be 3 or 7, received {kernel_size}"
            )

        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_feature = torch.mean(x, dim=1, keepdim=True)
        max_feature, _ = torch.max(x, dim=1, keepdim=True)

        pooled_feature = torch.cat(
            [avg_feature, max_feature],
            dim=1,
        )

        return self.sigmoid(self.conv(pooled_feature))


class ChannelAttention(nn.Module):
    """Generate channel attention with shape [B, C, 1, 1]."""

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
    ):
        super().__init__()

        if channels <= 0:
            raise ValueError(
                f"channels must be positive, received {channels}"
            )

        hidden_channels = max(channels // reduction, 8)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(
                channels,
                hidden_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                channels,
                kernel_size=1,
                bias=False,
            ),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.mlp(self.avg_pool(x)))


class AdaptiveFusion(nn.Module):
    """Learnable softmax fusion of spatial and channel attention."""

    def __init__(self):
        super().__init__()

        # Equal contribution at initialization.
        self.weight = nn.Parameter(
            torch.zeros(2, dtype=torch.float32)
        )

    def forward(
        self,
        spatial_attention: torch.Tensor,
        channel_attention: torch.Tensor,
    ) -> torch.Tensor:

        fusion_weight = torch.softmax(self.weight, dim=0)

        return (
            fusion_weight[0] * spatial_attention
            + fusion_weight[1] * channel_attention
        )


class AdaptiveSaliencyBlock(nn.Module):
    """
    Adaptive Saliency v2.

    Improvements:
        1. Spatial and channel attention
        2. Learnable softmax fusion
        3. Lightweight depthwise feature refinement
        4. Stable residual enhancement
        5. Identity-like initialization
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        spatial_kernel_size: int = 7,
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

        # Lightweight local feature refinement.
        self.refine = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.GroupNorm(32, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(32, channels),
        )

        # Small non-zero value allows attention branch to learn immediately,
        # while remaining close to the pretrained identity mapping.
        self.gamma = nn.Parameter(
            torch.tensor(0.01, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                "AdaptiveSaliencyBlock expects [B, C, H, W], "
                f"received {tuple(x.shape)}"
            )

        spatial_attention = self.spatial(x)
        channel_attention = self.channel(x)

        fused_attention = self.fusion(
            spatial_attention,
            channel_attention,
        )

        attended_feature = x * fused_attention
        refined_feature = self.refine(attended_feature)

        return x + self.gamma * refined_feature
