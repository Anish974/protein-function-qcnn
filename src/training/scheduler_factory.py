"""
Module
-------
Learning Rate Scheduler

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


class SchedulerFactory:
    """
    Scheduler construction.
    """

    @staticmethod
    def cosine(
        optimizer,
        epochs: int,
    ):

        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=epochs,
        )