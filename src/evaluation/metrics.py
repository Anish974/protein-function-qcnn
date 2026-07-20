"""
Module
-------
Evaluation Metrics

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
Computes standard classification metrics.
"""

from __future__ import annotations

from typing import Dict

from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import cohen_kappa_score


class Metrics:
    """
    Performance metric calculator.
    """

    @staticmethod
    def compute(
        targets,
        predictions,
    ) -> Dict[str, float]:

        return {

            "Accuracy":

                accuracy_score(
                    targets,
                    predictions,
                ),

            "Balanced Accuracy":

                balanced_accuracy_score(
                    targets,
                    predictions,
                ),

            "Precision":

                precision_score(
                    targets,
                    predictions,
                    average="weighted",
                    zero_division=0,
                ),

            "Recall":

                recall_score(
                    targets,
                    predictions,
                    average="weighted",
                    zero_division=0,
                ),

            "F1 Score":

                f1_score(
                    targets,
                    predictions,
                    average="weighted",
                    zero_division=0,
                ),

            "Matthews Correlation":

                matthews_corrcoef(
                    targets,
                    predictions,
                ),

            "Cohen Kappa":

                cohen_kappa_score(
                    targets,
                    predictions,
                ),

        }