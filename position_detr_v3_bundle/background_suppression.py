import torch
import torch.nn as nn


class BackgroundSuppressionGate(nn.Module):
    """
    Stable foreground enhancement and background suppression.

    The attention gate is in [0, 1]. It is remapped to [-1, 1]:

        signed_gate = 2 * gate - 1

    The output is:

        y = x * (1 + gamma * signed_gate)

    Therefore:
        gate < 0.5  -> background response is suppressed
        gate > 0.5  -> foreground response is enhanced

    gamma starts small so the module is close to identity at initialization.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        spatial_kernel_size: int = 7,
        gamma_init: float = 0.01,
    ):
        super().__init__()

        if channels <= 0:
            raise ValueError(f"channels must be positive, received {channels}")
        if reduction <= 0:
            raise ValueError(f"reduction must be positive, received {reduction}")
        if spatial_kernel_size not in (3, 7):
            raise ValueError(
                "spatial_kernel_size must be 3 or 7, "
                f"received {spatial_kernel_size}"
            )

        hidden_channels = max(channels // reduction, 8)

        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
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
            nn.Sigmoid(),
        )

        self.spatial_gate = nn.Sequential(
            nn.Conv2d(
                2,
                1,
                kernel_size=spatial_kernel_size,
                padding=spatial_kernel_size // 2,
                bias=False,
            ),
            nn.Sigmoid(),
        )

        # A scalar per block. A small positive value keeps the block
        # close to identity while allowing it to learn immediately.
        self.gamma = nn.Parameter(
            torch.tensor(float(gamma_init), dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                "BackgroundSuppressionGate expects [B, C, H, W], "
                f"received {tuple(x.shape)}"
            )

        channel_attention = self.channel_gate(x)

        avg_feature = torch.mean(x, dim=1, keepdim=True)
        max_feature, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attention = self.spatial_gate(
            torch.cat([avg_feature, max_feature], dim=1)
        )

        gate = channel_attention * spatial_attention
        signed_gate = 2.0 * gate - 1.0

        # Bound the effective strength to (-1, 1) for stability.
        strength = torch.tanh(self.gamma)

        return x * (1.0 + strength * signed_gate)
