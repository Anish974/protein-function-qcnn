"""
Module
-------
Angle Encoder

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
Transforms classical CNN features into valid quantum
rotation angles.
"""

from __future__ import annotations

import torch


class AngleEncoder:
    """
    Maps compressed CNN features into the range [-π, π].
    """

    def encode(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        features
            Shape:
            (Batch,10)

        Returns
        -------
        Rotation angles.
        """

        return torch.pi * torch.tanh(features)