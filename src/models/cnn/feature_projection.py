"""
Module
-------
Feature Projection

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
Projects pooled CNN representation into a compact
128-dimensional latent feature space.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FeatureProjection(nn.Module):
    """
    Dense projection network.
    """

    def __init__(
        self,
        input_features: int = 256,
        output_features: int = 128,
        dropout: float = 0.20,
    ) -> None:

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_features,
                256,
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(dropout),

            nn.Linear(
                256,
                output_features,
            ),

            nn.ReLU(inplace=True)

        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:

        return self.network(features)