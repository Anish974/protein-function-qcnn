"""
Module
-------
Optimizer Factory

Author
------
Amit Pimpalkar

Organization
------------
RBU, Nagpur

Year
----
2026
"""

from __future__ import annotations

import torch


class OptimizerFactory:
    """
    Builds optimizers.
    """

    @staticmethod
    def adamw(
        model,
        learning_rate: float,
        weight_decay: float,
    ):

        return torch.optim.AdamW(
            params=model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )