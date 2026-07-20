"""
Module
-------
Early Stopping

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
Stops training when validation loss no longer improves.
"""

from __future__ import annotations


class EarlyStopping:
    """
    Monitors validation loss.
    """

    def __init__(
        self,
        patience: int = 10,
        minimum_delta: float = 1e-4,
    ) -> None:

        self.patience = patience
        self.minimum_delta = minimum_delta

        self.best_loss = float("inf")
        self.counter = 0
        self.stop_training = False

    def update(
        self,
        validation_loss: float,
    ) -> bool:
        """
        Update stopping criterion.

        Returns
        -------
        True if training should stop.
        """

        if validation_loss < self.best_loss - self.minimum_delta:

            self.best_loss = validation_loss
            self.counter = 0

        else:

            self.counter += 1

            if self.counter >= self.patience:

                self.stop_training = True

        return self.stop_training