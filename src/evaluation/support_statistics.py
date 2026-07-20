"""
Module
-------
Support Statistics

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
Provides dataset support statistics.
"""

from __future__ import annotations

import pandas as pd


class SupportStatistics:
    """
    Computes sample distribution.
    """

    @staticmethod
    def summarize(
        class_table: pd.DataFrame,
    ) -> pd.DataFrame:

        summary = class_table.copy()

        summary["Support Percentage"] = (

            summary["Support"]

            / summary["Support"].sum()

        ) * 100.0

        return summary