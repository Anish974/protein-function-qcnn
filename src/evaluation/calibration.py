"""
Module
-------
Calibration Analysis

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
Computes Expected Calibration Error (ECE),
Maximum Calibration Error (MCE), and
Brier Score for multi-class classification.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import brier_score_loss


class CalibrationAnalysis:
    """
    Computes calibration statistics.
    """

    @staticmethod
    def expected_calibration_error(
        probabilities: np.ndarray,
        targets: np.ndarray,
        bins: int = 10,
    ) -> Dict[str, float]:

        confidences = np.max(probabilities, axis=1)
        predictions = np.argmax(probabilities, axis=1)

        accuracy = predictions == targets

        boundaries = np.linspace(0.0, 1.0, bins + 1)

        ece = 0.0
        mce = 0.0

        for index in range(bins):

            lower = boundaries[index]
            upper = boundaries[index + 1]

            mask = (confidences > lower) & (confidences <= upper)

            if np.sum(mask) == 0:
                continue

            bin_accuracy = np.mean(accuracy[mask])
            bin_confidence = np.mean(confidences[mask])

            gap = abs(bin_accuracy - bin_confidence)

            ece += gap * np.mean(mask)

            if gap > mce:
                mce = gap

        return {

            "ECE": float(ece),

            "MCE": float(mce),

        }

    @staticmethod
    def multiclass_brier_score(
        probabilities: np.ndarray,
        targets: np.ndarray,
    ) -> float:

        classes = probabilities.shape[1]

        scores = []

        for class_index in range(classes):

            binary_targets = (
                targets == class_index
            ).astype(float)

            scores.append(

                brier_score_loss(

                    binary_targets,

                    probabilities[:, class_index],

                )

            )

        return float(np.mean(scores))