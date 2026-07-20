"""
Module
-------
Ablation Experiment

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
Defines the common interface implemented by every
ablation experiment.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict


class AblationExperiment(ABC):
    """
    Base class for all ablation experiments.
    """

    def __init__(
        self,
        configuration: Dict[str, Any],
    ) -> None:

        self.configuration = configuration

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Returns experiment name.
        """

    @abstractmethod
    def configure_model(
        self,
    ):
        """
        Configure the model variant.
        """

    @abstractmethod
    def train(
        self,
    ):
        """
        Execute model training.
        """

    @abstractmethod
    def evaluate(
        self,
    ) -> Dict[str, float]:
        """
        Evaluate trained model.
        """

    def execute(
        self,
    ) -> Dict[str, float]:
        """
        Executes a complete ablation experiment.
        """

        self.configure_model()

        self.train()

        return self.evaluate()