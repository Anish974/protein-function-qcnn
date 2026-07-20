"""
Module
-------
Metrics Export

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
Exports evaluation results.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class MetricsExporter:
    """
    Saves evaluation tables.
    """

    def __init__(
        self,
        output_directory: str,
    ) -> None:

        self.output_directory = Path(output_directory)

        self.output_directory.mkdir(

            parents=True,

            exist_ok=True,

        )

    def export_csv(

        self,

        dataframe: pd.DataFrame,

        filename: str,

    ) -> None:

        dataframe.to_csv(

            self.output_directory / filename,

            index=False,

        )

    def export_excel(

        self,

        dataframe: pd.DataFrame,

        filename: str,

    ) -> None:

        dataframe.to_excel(

            self.output_directory / filename,

            index=False,

        )