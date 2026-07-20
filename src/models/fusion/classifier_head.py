"""
Module
-------
Classification Head

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
Maps fused hybrid features into class probabilities.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """
    Multi-class classifier.
    """

    def __init__(
        self,
        input_dimension: int = 128,
        number_of_classes: int = 10,
    ) -> None:

        super().__init__()

        self.classifier = nn.Sequential(

            nn.Linear(
                input_dimension,
                64,
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(0.20),

            nn.Linear(
                64,
                number_of_classes,
            ),

        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:

        return self.classifier(features)