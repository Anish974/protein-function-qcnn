"""
Module
-------
Multi-Class ROC Analysis

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
Computes One-vs-Rest ROC curves and Area Under the Curve
(AUC) scores for multi-class protein sequence classification.

Usage
-----
This module expects probability scores produced by the HQNN
classifier together with integer encoded target labels.
"""

from __future__ import annotations

from typing import Dict
from typing import List

import numpy as np

from sklearn.preprocessing import label_binarize

from sklearn.metrics import (
    roc_curve,
    auc,
)


class ROCAnalysis:
    """
    Computes multi-class ROC statistics.
    """

    @staticmethod
    def compute(
        targets: List[int],
        probabilities: np.ndarray,
        number_of_classes: int,
    ) -> Dict:

        targets = np.asarray(targets)

        binary_targets = label_binarize(

            targets,

            classes=list(range(number_of_classes))

        )

        false_positive_rate = {}

        true_positive_rate = {}

        auc_score = {}

        for class_index in range(number_of_classes):

            fpr, tpr, _ = roc_curve(

                binary_targets[:, class_index],

                probabilities[:, class_index]

            )

            false_positive_rate[class_index] = fpr

            true_positive_rate[class_index] = tpr

            auc_score[class_index] = auc(

                fpr,

                tpr

            )

        return {

            "fpr": false_positive_rate,

            "tpr": true_positive_rate,

            "auc": auc_score,

        }