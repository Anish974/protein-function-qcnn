"""
Module
-------
Multi-scale CNN Block

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
Extracts sequence motifs at multiple biological scales.

Kernel Sizes
------------
3
5
7
11
"""

import torch
import torch.nn as nn

from .conv_block import ConvolutionBlock


class MultiScaleBlock(nn.Module):
    """
    Parallel multi-scale convolution.
    """

    def __init__(
        self,
        embedding_dimension: int = 64,
        filters_per_branch: int = 64,
    ) -> None:

        super().__init__()

        self.kernel3 = ConvolutionBlock(
            embedding_dimension,
            filters_per_branch,
            3,
        )

        self.kernel5 = ConvolutionBlock(
            embedding_dimension,
            filters_per_branch,
            5,
        )

        self.kernel7 = ConvolutionBlock(
            embedding_dimension,
            filters_per_branch,
            7,
        )

        self.kernel11 = ConvolutionBlock(
            embedding_dimension,
            filters_per_branch,
            11,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args
        ----
        x
            Shape
            (Batch, Embedding, Length)

        Returns
        -------
        Tensor

            Shape

            (Batch,256,Length)
        """

        feature3 = self.kernel3(x)

        feature5 = self.kernel5(x)

        feature7 = self.kernel7(x)

        feature11 = self.kernel11(x)

        return torch.cat(

            (

                feature3,

                feature5,

                feature7,

                feature11,

            ),

            dim=1,

        )