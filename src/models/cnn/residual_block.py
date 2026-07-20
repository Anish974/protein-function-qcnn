"""
Module
-------
Residual Refinement Block

Author
------
Amit Pimpalkar

Organization
------------
RBU, Nagpur

Year
----
2026

Purpose
-------
Refines extracted convolutional features while preserving
gradient flow through residual learning.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Standard residual feature refinement block.
    """

    def __init__(self, channels: int) -> None:

        super().__init__()

        self.layers = nn.Sequential(

            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm1d(channels),

            nn.ReLU(inplace=True),

            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm1d(channels)

        )

        self.activation = nn.ReLU(inplace=True)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        identity = x

        output = self.layers(x)

        output = output + identity

        return self.activation(output)