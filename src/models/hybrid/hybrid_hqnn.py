"""
Module
-------
Hybrid Quantum-Classical Protein Classifier

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
End-to-end Hybrid Quantum-Classical Network for
multi-class protein sequence classification.

Architecture
------------
Protein Sequence

↓

CNN Encoder

↓

128-D Classical Feature

↓

10-D Quantum Feature

↓

Variational Quantum Classifier

↓

Feature Fusion

↓

Classification Head
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from src.models.cnn.cnn_encoder import CNNEncoder
from src.models.quantum.quantum_classifier import (
    VariationalQuantumClassifier,
)
from src.models.fusion.feature_fusion import FeatureFusion
from src.models.fusion.classifier_head import ClassificationHead


class HybridHQNN(nn.Module):
    """
    Complete HQNN architecture.
    """

    def __init__(
        self,
        number_of_classes: int = 10,
    ) -> None:

        super().__init__()

        self.encoder = CNNEncoder()

        self.quantum = VariationalQuantumClassifier()

        self.fusion = FeatureFusion()

        self.classifier = ClassificationHead(
            number_of_classes=number_of_classes
        )

    def forward(
        self,
        sequence: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward propagation.

        Args:
            sequence:
                Shape (Batch,300)

        Returns
        -------
        Dictionary containing
        logits,
        probabilities,
        predicted labels,
        intermediate embeddings.
        """

        classical_features, quantum_input = self.encoder(
            sequence
        )

        quantum_embedding = self.quantum(
            quantum_input
        )

        fused_features = self.fusion(

            classical_features,

            quantum_embedding,

        )

        logits = self.classifier(

            fused_features

        )

        probabilities = torch.softmax(

            logits,

            dim=1,

        )

        predictions = torch.argmax(

            probabilities,

            dim=1,

        )

        return {

            "logits": logits,

            "probabilities": probabilities,

            "predictions": predictions,

            "classical_features": classical_features,

            "quantum_features": quantum_embedding,

            "hybrid_features": fused_features,

        }