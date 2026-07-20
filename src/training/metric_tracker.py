"""
Module
-------
Metric Tracker

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
Accumulates predictions and computes
classification metrics.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

)


class MetricTracker:
    """
    Computes classification metrics.
    """

    def evaluate(
        self,
        targets,
        predictions,
    ) -> Dict[str, float]:

        targets = np.asarray(targets)

        predictions = np.asarray(predictions)

        return {

            "accuracy":

                accuracy_score(
                    targets,
                    predictions,
                ),

            "precision":

                precision_score(
                    targets,
                    predictions,
                    average="weighted",
                    zero_division=0,
                ),

            "recall":

                recall_score(
                    targets,
                    predictions,
                    average="weighted",
                    zero_division=0,
                ),

            "f1":

                f1_score(
                    targets,
                    predictions,
                    average="weighted",
                    zero_division=0,
                ),

        }