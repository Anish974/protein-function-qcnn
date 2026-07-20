"""
Module
-------
Calibration Export

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

from pathlib import Path

import pandas as pd


class CalibrationExporter:
    """
    Saves calibration summaries.
    """

    def __init__(
        self,
        directory: str,
    ):

        self.directory = Path(directory)

        self.directory.mkdir(

            parents=True,

            exist_ok=True,

        )

    def save(

        self,

        dataframe: pd.DataFrame,

        filename: str = "calibration_summary.csv",

    ):

        dataframe.to_csv(

            self.directory / filename,

            index=False,

        )

        dataframe.to_excel(

            self.directory /

            filename.replace(".csv", ".xlsx"),

            index=False,

        )