"""
Module
-------
Bootstrap Estimation

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
Bootstrap estimation of evaluation metrics.
"""

from __future__ import annotations

from typing import Callable
from typing import Dict

import logging
import numpy as np

LOGGER = logging.getLogger(__name__)


class BootstrapEstimator:
    """
    Performs bootstrap resampling.
    """

    def __init__(
        self,
        iterations: int = 1000,
        confidence_level: float = 0.95,
        random_state: int = 42,
    ) -> None:

        self.iterations = iterations
        self.confidence_level = confidence_level
        self.random_state = random_state

    def estimate(
        self,
        targets: np.ndarray,
        predictions: np.ndarray,
        metric_function: Callable,
    ) -> Dict[str, float]:
        """
        Estimate bootstrap confidence interval.
        """

        rng = np.random.default_rng(self.random_state)

        scores = []

        sample_size = len(targets)

        for _ in range(self.iterations):

            indices = rng.integers(
                0,
                sample_size,
                sample_size,
            )

            score = metric_function(

                targets[indices],

                predictions[indices],

            )

            scores.append(score)

        scores = np.asarray(scores)

        alpha = 1.0 - self.confidence_level

        lower = np.percentile(
            scores,
            100 * alpha / 2,
        )

        upper = np.percentile(
            scores,
            100 * (1 - alpha / 2),
        )

        return {

            "mean": float(scores.mean()),

            "std": float(scores.std(ddof=1)),

            "lower": float(lower),

            "upper": float(upper),

        }