"""
Module
-------
Ablation Registry

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
Maintains available ablation experiments.
"""

from __future__ import annotations

from typing import Dict
from typing import Type

from .experiment import AblationExperiment


class AblationRegistry:
    """
    Stores experiment classes.
    """

    def __init__(self):

        self._registry: Dict[
            str,
            Type[AblationExperiment]
        ] = {}

    def register(
        self,
        experiment: Type[AblationExperiment],
    ) -> None:

        self._registry[
            experiment.__name__
        ] = experiment

    def get(
        self,
        name: str,
    ) -> Type[AblationExperiment]:

        return self._registry[name]

    def experiments(self):

        return self._registry.items()