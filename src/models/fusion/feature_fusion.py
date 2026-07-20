"""
Module
-------
Feature Fusion Network

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
Fuses classical CNN embeddings with quantum embeddings
to form a unified latent representation.

The fusion network learns complementary information
instead of relying on simple concatenation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FeatureFusion(nn.Module):
    """
    Hybrid feature fusion module.
    """

    def __init__(
        self,
        classical_dimension: int = 128,
        quantum_dimension: int = 10,
        fused_dimension: int = 128,
        dropout: float = 0.30,
    ) -> None:

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                classical_dimension + quantum_dimension,
                256,
            ),

            nn.BatchNorm1d(256),

            nn.ReLU(inplace=True),

            nn.Dropout(dropout),

            nn.Linear(
                256,
                fused_dimension,
            ),

            nn.ReLU(inplace=True),

        )

    def forward(
        self,
        classical_features: torch.Tensor,
        quantum_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse classical and quantum features.

        Args:
            classical_features:
                Shape (Batch,128)

            quantum_features:
                Shape (Batch,10)

        Returns
        -------
        Tensor
            Shape (Batch,128)
        """

        fused = torch.cat(

            [

                classical_features,

                quantum_features,

            ],

            dim=1,

        )

        return self.network(fused)