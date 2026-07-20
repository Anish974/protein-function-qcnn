"""
Module
-------
Per-Class Metrics

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
Computes class-wise evaluation metrics.
"""

from __future__ import annotations

import pandas as pd

from sklearn.metrics import precision_recall_fscore_support


class PerClassMetrics:
    """
    Computes metrics for every protein family.
    """

    @staticmethod
    def evaluate(
        targets,
        predictions,
        class_names,
    ) -> pd.DataFrame:

        precision, recall, f1, support = \
            precision_recall_fscore_support(

                targets,

                predictions,

                zero_division=0,

            )

        dataframe = pd.DataFrame({

            "Class": class_names,

            "Precision": precision,

            "Recall": recall,

            "F1 Score": f1,

            "Support": support,

        })

        return dataframe