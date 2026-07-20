"""
Module
-------
Loss Functions

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
Centralized loss construction.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class LossFactory:
    """
    Creates loss functions used during training.
    """

    @staticmethod
    def weighted_cross_entropy(
        class_weights: Optional[torch.Tensor] = None,
    ) -> nn.Module:
        """
        Build weighted cross entropy loss.

        Args:
            class_weights:
                Tensor containing class weights.

        Returns
        -------
        CrossEntropyLoss
        """

        return nn.CrossEntropyLoss(
            weight=class_weights
        )