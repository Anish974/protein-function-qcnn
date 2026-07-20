"""
Module
-------
Quantum Feature Compressor

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
Compresses classical CNN features into the
10-dimensional representation consumed by the
Variational Quantum Classifier.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class QuantumFeatureCompressor(nn.Module):
    """
    Compress 128 classical features into
    ten quantum rotation angles.
    """

    def __init__(
        self,
        input_dimension: int = 128,
        quantum_dimension: int = 10,
    ) -> None:

        super().__init__()

        self.layers = nn.Sequential(

            nn.Linear(
                input_dimension,
                64,
            ),

            nn.ReLU(inplace=True),

            nn.Linear(
                64,
                quantum_dimension,
            ),

            nn.Tanh()

        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:

        return self.layers(features)