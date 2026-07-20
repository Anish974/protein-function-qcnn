"""
Module
-------
Ablation Results

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
Stores and exports ablation experiment results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict
from typing import List

import pandas as pd


class AblationResults:
    """
    Collects experiment metrics.
    """

    def __init__(
        self,
        output_directory: str,
    ) -> None:

        self.output_directory = Path(
            output_directory
        )

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.records: List[Dict] = []

    def add(
        self,
        experiment_name: str,
        metrics: Dict[str, float],
    ) -> None:

        record = {

            "Experiment":

                experiment_name,

            **metrics,

        }

        self.records.append(record)

    def export(
        self,
    ) -> None:

        dataframe = pd.DataFrame(
            self.records
        )

        dataframe.to_csv(

            self.output_directory
            / "ablation_summary.csv",

            index=False,

        )

        dataframe.to_excel(

            self.output_directory
            / "ablation_summary.xlsx",

            index=False,

        )