"""
ROC-AUC Evaluation
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import roc_auc_score


class ROCAUC:
    """
    Computes macro ROC-AUC.
    """

    @staticmethod
    def compute(
        y_true: np.ndarray,
        y_score: np.ndarray,
    ) -> Dict[str, float]:

        score = roc_auc_score(

            y_true,

            y_score,

            average="macro",

        )

        return {

            "AUC": float(score)

        }