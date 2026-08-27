from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _valid_group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class AdaptiveMultiScaleFusion(nn.Module):
    """
    Fuse P2, P3 and P4 into P3 resolution.

    Input:
        features = [P2, P3, P4]

    Output:
        A P3-shaped tensor with the same channel count as P3.

    Each level is projected to P3 channels and resized to P3 spatial size.
    Learned softmax weights determine the contribution of each level.
    A small residual gamma keeps the module near identity initially.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        target_index: int = 1,
        gamma_init: float = 0.01,
    ):
        super().__init__()

        if len(in_channels) != 3:
            raise ValueError(
                "AdaptiveMultiScaleFusion expects exactly three levels "
                f"[P2, P3, P4], received {len(in_channels)}"
            )
        if not 0 <= target_index < len(in_channels):
            raise ValueError(
                f"target_index must be in [0, {len(in_channels) - 1}], "
                f"received {target_index}"
            )

        self.target_index = target_index
        out_channels = int(in_channels[target_index])
        groups = _valid_group_count(out_channels)

        self.projections = nn.ModuleList()
        for channels in in_channels:
            if int(channels) == out_channels:
                projection = nn.Identity()
            else:
                projection = nn.Sequential(
                    nn.Conv2d(
                        int(channels),
                        out_channels,
                        kernel_size=1,
                        bias=False,
                    ),
                    nn.GroupNorm(groups, out_channels),
                )
            self.projections.append(projection)

        # Equal scale contribution at initialization.
        self.scale_logits = nn.Parameter(
            torch.zeros(len(in_channels), dtype=torch.float32)
        )

        self.refine = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                groups=out_channels,
                bias=False,
            ),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(groups, out_channels),
        )

        self.gamma = nn.Parameter(
            torch.tensor(float(gamma_init), dtype=torch.float32)
        )

    def forward(self, features):
        if len(features) != len(self.projections):
            raise ValueError(
                f"Expected {len(self.projections)} features, "
                f"received {len(features)}"
            )

        target_feature = features[self.target_index]
        target_size = target_feature.shape[-2:]
        weights = torch.softmax(self.scale_logits, dim=0)

        fused = None
        for weight, projection, feature in zip(
            weights,
            self.projections,
            features,
        ):
            projected = projection(feature)

            if projected.shape[-2:] != target_size:
                projected = F.interpolate(
                    projected,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )

            weighted_feature = weight * projected
            fused = (
                weighted_feature
                if fused is None
                else fused + weighted_feature
            )

        refined = self.refine(fused)
        strength = torch.tanh(self.gamma)

        # P3 identity residual path.
        return target_feature + strength * refined
