"""
Module
-------
Statistical Summary

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
Creates publication-ready statistical tables.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class StatisticalSummary:
    """
    Consolidates statistical results.
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

    def export(
        self,
        summary: dict,
        filename: str = "statistical_summary.csv",
    ) -> None:

        dataframe = pd.DataFrame(

            [

                {

                    "Statistic": key,

                    "Value": value,

                }

                for key, value in summary.items()

            ]

        )

        csv_path = self.output_directory / filename

        xlsx_path = csv_path.with_suffix(".xlsx")

        dataframe.to_csv(

            csv_path,

            index=False,

        )

        dataframe.to_excel(

            xlsx_path,

            index=False,

        )