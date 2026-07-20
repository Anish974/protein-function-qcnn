"""
Module
-------
Prediction Collector

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
Collects predictions from the trained HQNN model.
"""

from __future__ import annotations

from typing import Dict
from typing import List

import torch


class PredictionCollector:
    """
    Runs inference on the evaluation dataset.
    """

    def __init__(
        self,
        model,
        device,
    ) -> None:

        self.model = model
        self.device = device

    @torch.no_grad()
    def collect(
        self,
        dataloader,
    ) -> Dict[str, List]:

        self.model.eval()

        targets = []
        predictions = []
        probabilities = []

        for sequences, labels in dataloader:

            sequences = sequences.to(self.device)

            outputs = self.model(sequences)

            logits = outputs["logits"]

            probability = torch.softmax(
                logits,
                dim=1,
            )

            prediction = torch.argmax(
                probability,
                dim=1,
            )

            targets.extend(labels.numpy().tolist())

            predictions.extend(
                prediction.cpu().numpy().tolist()
            )

            probabilities.extend(
                probability.cpu().numpy().tolist()
            )

        return {

            "targets": targets,

            "predictions": predictions,

            "probabilities": probabilities,

        }