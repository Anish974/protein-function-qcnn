"""
Module
-------
Confusion Matrix

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
Computes raw and normalized confusion matrices for
multi-class protein sequence classification.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import confusion_matrix


class ConfusionMatrixEvaluator:
    """
    Computes confusion matrices.
    """

    @staticmethod
    def compute(
        targets,
        predictions,
    ) -> Dict[str, np.ndarray]:
        """
        Compute raw, row-normalized and column-normalized
        confusion matrices.

        Args:
            targets:
                Ground truth labels.

            predictions:
                Predicted labels.

        Returns
        -------
        Dictionary of confusion matrices.
        """

        matrix = confusion_matrix(
            targets,
            predictions,
        )

        row_sum = matrix.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1

        column_sum = matrix.sum(axis=0, keepdims=True)
        column_sum[column_sum == 0] = 1

        row_normalized = matrix / row_sum
        column_normalized = matrix / column_sum

        return {

            "raw": matrix,

            "row_normalized": row_normalized,

            "column_normalized": column_normalized,

        }