"""
Module
-------
Complete CNN Encoder

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
Hybrid classical feature extractor.

Architecture
------------
Embedding
        ↓
Multi-scale CNN
        ↓
Residual Refinement
        ↓
Adaptive Pooling
        ↓
Dense Projection
        ↓
128-D Feature Vector
        ↓
Quantum Compressor
        ↓
10-D Quantum Feature Vector
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from src.models.embeddings.embedding_layer import ProteinEmbedding
from src.models.cnn.multi_scale_block import MultiScaleBlock
from src.models.cnn.global_pool import GlobalPooling
from src.models.cnn.residual_block import ResidualBlock
from src.models.cnn.feature_projection import FeatureProjection
from src.models.cnn.feature_compressor import QuantumFeatureCompressor


class CNNEncoder(nn.Module):
    """
    Complete classical encoder.
    """

    def __init__(self) -> None:

        super().__init__()

        self.embedding = ProteinEmbedding()

        self.multiscale = MultiScaleBlock()

        self.refinement = ResidualBlock(256)

        self.pool = GlobalPooling()

        self.projection = FeatureProjection()

        self.quantum_projection = QuantumFeatureCompressor()

    def forward(
        self,
        sequence: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        sequence
            Shape:
            (Batch,300)

        Returns
        -------
        classical_features
            Shape:
            (Batch,128)

        quantum_features
            Shape:
            (Batch,10)
        """

        embedding = self.embedding(sequence)

        embedding = embedding.transpose(1, 2)

        features = self.multiscale(embedding)

        features = self.refinement(features)

        pooled = self.pool(features)

        classical_features = self.projection(pooled)

        quantum_features = self.quantum_projection(
            classical_features
        )

        return (
            classical_features,
            quantum_features,
        )