"""
Module
-------
Calibration Metrics Summary

Author
------
Amit Pimpalkar

Organization
------------
RBU, Nagpur

Year
----
2026
"""

from __future__ import annotations

import pandas as pd


class CalibrationMetrics:
    """
    Creates calibration summary table.
    """

    @staticmethod
    def create(
        ece: float,
        mce: float,
        brier: float,
    ) -> pd.DataFrame:

        return pd.DataFrame(

            {

                "Metric": [

                    "Expected Calibration Error",

                    "Maximum Calibration Error",

                    "Brier Score",

                ],

                "Value": [

                    ece,

                    mce,

                    brier,

                ],

            }

        )