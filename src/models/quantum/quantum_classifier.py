"""
Module
-------
Variational Quantum Classifier

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
Transforms CNN embeddings using a trainable
variational quantum circuit.

The output is fused with classical features
before final classification.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .quantum_layer import QuantumLayer


class VariationalQuantumClassifier(nn.Module):
    """
    Quantum feature extractor.
    """

    def __init__(self):

        super().__init__()

        self.layer = QuantumLayer()

    def forward(
        self,
        quantum_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        quantum_features

            Shape

            (Batch,10)

        Returns
        -------
        Quantum embedding

            Shape

            (Batch,10)
        """

        return self.layer(

            quantum_features

        )