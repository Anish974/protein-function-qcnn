"""
Module
-------
Protein Embedding Layer

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
Transforms integer encoded amino acids into dense feature vectors.

Architecture
------------
Vocabulary Size : 21
Embedding Size  : 64
"""

from typing import Optional

import torch
import torch.nn as nn


class ProteinEmbedding(nn.Module):
    """
    Learnable amino acid embedding.
    """

    def __init__(
        self,
        vocabulary_size: int = 21,
        embedding_dimension: int = 64,
        padding_index: int = 0,
    ) -> None:

        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_dimension,
            padding_idx=padding_index,
        )

    def forward(
        self,
        sequence: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert amino acid indices into embeddings.

        Args:
            sequence:
                Tensor of shape (Batch, Length)

        Returns
        -------
        Tensor
            Shape:
            (Batch, Length, Embedding Dimension)
        """

        return self.embedding(sequence)