"""
Module
-------
Convolution Block

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
Single convolutional feature extraction block.
"""

from typing import Tuple

import torch
import torch.nn as nn


class ConvolutionBlock(nn.Module):
    """
    Standard convolution block.

    Conv1D
        ↓
    BatchNorm
        ↓
    ReLU
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
    ) -> None:

        super().__init__()

        padding = kernel_size // 2

        self.block = nn.Sequential(

            nn.Conv1d(

                in_channels=input_channels,

                out_channels=output_channels,

                kernel_size=kernel_size,

                padding=padding,

            ),

            nn.BatchNorm1d(output_channels),

            nn.ReLU(inplace=True),

        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.block(x)