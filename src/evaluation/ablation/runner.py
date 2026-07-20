"""
Module
-------
Ablation Runner

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
Executes all registered ablation experiments.
"""

from __future__ import annotations

import logging
from typing import Any
from typing import Dict

from .registry import AblationRegistry
from .results import AblationResults

LOGGER = logging.getLogger(__name__)


class AblationRunner:
    """
    Runs all ablation experiments.
    """

    def __init__(
        self,
        registry: AblationRegistry,
        output_directory: str,
    ) -> None:

        self.registry = registry

        self.results = AblationResults(
            output_directory
        )

    def execute(
        self,
        configuration: Dict[str, Any],
    ) -> None:

        for _, experiment_class in (
            self.registry.experiments()
        ):

            LOGGER.info(
                "Running %s",
                experiment_class.__name__,
            )

            experiment = experiment_class(
                configuration
            )

            metrics = experiment.execute()

            self.results.add(

                experiment.name,

                metrics,

            )

        self.results.export()


class AblationRunner:
    """
    Executes all ablation experiments.
    """

    def __init__(
        self,
        trainer,
        evaluator,
        configuration,
    ):

        self.trainer = trainer
        self.evaluator = evaluator

        self.builder = HybridModelBuilder(
            configuration
        )

    def execute(self):

        results = []

        for variant in ABLATION_VARIANTS:

            model = self.builder.build(
                variant
            )

            self.trainer.train(model)

            metrics = self.evaluator.evaluate(
                model
            )

            metrics["Model"] = variant.name

            results.append(metrics)

        return results