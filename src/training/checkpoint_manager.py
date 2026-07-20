"""
Module
-------
Checkpoint Manager

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
Stores and restores training checkpoints.
"""

from __future__ import annotations

from pathlib import Path

import torch


class CheckpointManager:
    """
    Handles model persistence.
    """

    def __init__(
        self,
        directory: str = "checkpoints",
    ) -> None:

        self.directory = Path(directory)

        self.directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save(
        self,
        filename: str,
        model,
        optimizer,
        epoch: int,
        validation_loss: float,
    ) -> None:

        checkpoint = {

            "epoch": epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "validation_loss":
                validation_loss,

        }

        torch.save(

            checkpoint,

            self.directory / filename,

        )

    def load(
        self,
        filename: str,
        model,
        optimizer=None,
    ):

        checkpoint = torch.load(
            self.directory / filename,
            map_location="cpu",
        )

        model.load_state_dict(

            checkpoint["model_state_dict"]

        )

        if optimizer is not None:

            optimizer.load_state_dict(

                checkpoint["optimizer_state_dict"]

            )

        return checkpoint