"""
Module
-------
Fmax Evaluation

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
Computes the maximum F-score (Fmax) for protein function
prediction across different probability thresholds.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import precision_recall_curve


class FMax:
    """
    Computes Fmax for multilabel protein prediction.
    """

    @staticmethod
    def compute(
        y_true: np.ndarray,
        y_score: np.ndarray,
    ) -> Dict[str, float]:

        best_f = 0.0
        best_threshold = 0.5

        for class_index in range(y_true.shape[1]):

            precision, recall, thresholds = precision_recall_curve(
                y_true[:, class_index],
                y_score[:, class_index],
            )

            fscore = (
                2 * precision * recall
            ) / (
                precision + recall + 1e-12
            )

            index = np.argmax(fscore)

            if fscore[index] > best_f:

                best_f = float(fscore[index])

                if index < len(thresholds):

                    best_threshold = float(
                        thresholds[index]
                    )

        return {

            "Fmax": best_f,

            "Threshold": best_threshold,

        }