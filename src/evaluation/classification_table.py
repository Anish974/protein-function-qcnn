"""
Module
-------
Classification Summary Table

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
Creates a summary table.
"""

from __future__ import annotations

import pandas as pd


class ClassificationTable:
    """
    Builds evaluation summary.
    """

    @staticmethod
    def create(
        metrics: dict,
    ) -> pd.DataFrame:

        dataframe = pd.DataFrame(

            {

                "Metric": list(metrics.keys()),

                "Value": list(metrics.values()),

            }

        )

        return dataframe