"""
Module
-------
Evaluation Engine

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
Coordinates complete model evaluation.
"""

from __future__ import annotations

from .prediction_collector import PredictionCollector
from .metrics import Metrics
from .report_generator import ReportGenerator


class Evaluator:
    """
    Central evaluation pipeline.
    """

    def __init__(
        self,
        model,
        device,
    ) -> None:

        self.collector = PredictionCollector(
            model,
            device,
        )

    def evaluate(
        self,
        dataloader,
        class_names,
    ):

        results = self.collector.collect(
            dataloader
        )

        metrics = Metrics.compute(

            results["targets"],

            results["predictions"],

        )

        report = ReportGenerator.create(

            results["targets"],

            results["predictions"],

            class_names,

        )

        return {

            "metrics": metrics,

            "classification_report": report,

            "targets": results["targets"],

            "predictions": results["predictions"],

            "probabilities": results["probabilities"],

        }