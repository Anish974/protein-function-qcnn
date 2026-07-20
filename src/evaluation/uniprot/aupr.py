"""
Area Under Precision Recall Curve
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import average_precision_score


class AUPR:
    """
    Computes macro AUPR.
    """

    @staticmethod
    def compute(
        y_true: np.ndarray,
        y_score: np.ndarray,
    ) -> Dict[str, float]:

        score = average_precision_score(

            y_true,

            y_score,

            average="macro",

        )

        return {

            "AUPR": float(score)

        }