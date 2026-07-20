"""
Module
-------
Training Engine

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
Implements supervised training and validation for the
Hybrid Quantum-Classical Protein Classification model.

Features
--------
* Automatic mixed precision
* Gradient clipping
* Metric aggregation
* Scheduler support
* GPU compatible
"""

from __future__ import annotations

from typing import Dict

import logging

import torch

from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast


LOGGER = logging.getLogger(__name__)


class Trainer:
    """
    Model trainer.
    """

    def __init__(

        self,

        model,

        optimizer,

        criterion,

        scheduler,

        device,

        gradient_clip: float = 1.0,

    ):

        self.model = model

        self.optimizer = optimizer

        self.criterion = criterion

        self.scheduler = scheduler

        self.device = device

        self.gradient_clip = gradient_clip

        self.scaler = GradScaler()

    def train_epoch(
        self,
        dataloader,
    ) -> Dict[str, float]:
        """
        Train one epoch.
        """

        self.model.train()

        running_loss = 0.0

        correct = 0

        total = 0

        for sequences, labels in dataloader:

            sequences = sequences.to(self.device)

            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            with autocast():

                outputs = self.model(sequences)

                logits = outputs["logits"]

                loss = self.criterion(
                    logits,
                    labels,
                )

            self.scaler.scale(loss).backward()

            self.scaler.unscale_(self.optimizer)

            torch.nn.utils.clip_grad_norm_(

                self.model.parameters(),

                self.gradient_clip,

            )

            self.scaler.step(self.optimizer)

            self.scaler.update()

            predictions = torch.argmax(

                logits,

                dim=1,

            )

            running_loss += loss.item()

            total += labels.size(0)

            correct += (

                predictions == labels

            ).sum().item()

        if self.scheduler is not None:

            self.scheduler.step()

        return {

            "loss":

                running_loss / len(dataloader),

            "accuracy":

                correct / total,

        }

    @torch.no_grad()
    def validate(
        self,
        dataloader,
    ) -> Dict[str, float]:
        """
        Evaluate one epoch.
        """

        self.model.eval()

        running_loss = 0.0

        correct = 0

        total = 0

        for sequences, labels in dataloader:

            sequences = sequences.to(self.device)

            labels = labels.to(self.device)

            outputs = self.model(

                sequences

            )

            logits = outputs["logits"]

            loss = self.criterion(

                logits,

                labels,

            )

            predictions = torch.argmax(

                logits,

                dim=1,

            )

            running_loss += loss.item()

            total += labels.size(0)

            correct += (

                predictions == labels

            ).sum().item()

        return {

            "loss":

                running_loss / len(dataloader),

            "accuracy":

                correct / total,

        }

    def fit(
        self,
        train_loader,
        validation_loader,
        epochs: int,
    ):
        """
        Execute training.

        Returns
        -------
        History dictionary.
        """

        history = {

            "train_loss": [],

            "train_accuracy": [],

            "validation_loss": [],

            "validation_accuracy": [],

        }

        LOGGER.info("Training started.")

        for epoch in range(epochs):

            train_metrics = self.train_epoch(

                train_loader

            )

            validation_metrics = self.validate(

                validation_loader

            )

            history["train_loss"].append(

                train_metrics["loss"]

            )

            history["train_accuracy"].append(

                train_metrics["accuracy"]

            )

            history["validation_loss"].append(

                validation_metrics["loss"]

            )

            history["validation_accuracy"].append(

                validation_metrics["accuracy"]

            )

            LOGGER.info(

                "Epoch %03d | "
                "Train Loss %.4f | "
                "Validation Loss %.4f | "
                "Train Accuracy %.4f | "
                "Validation Accuracy %.4f",

                epoch + 1,

                train_metrics["loss"],

                validation_metrics["loss"],

                train_metrics["accuracy"],

                validation_metrics["accuracy"],

            )

        LOGGER.info("Training completed.")

        return history