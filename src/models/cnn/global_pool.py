"""
Module
-------
Global Pooling

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
Converts variable-length feature maps into
fixed-dimensional feature vectors.
"""

import torch
import torch.nn as nn


class GlobalPooling(nn.Module):
    """
    Adaptive max pooling.
    """

    def __init__(self):

        super().__init__()

        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:

        pooled = self.pool(features)

        return pooled.squeeze(-1)