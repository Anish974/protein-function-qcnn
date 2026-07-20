"""
Module
-------
Training History Logger

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
Stores training history and exports
CSV summaries.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class HistoryLogger:
    """
    Records training history.
    """

    def __init__(
        self,
        output_directory: str = "experiments",
    ) -> None:

        self.output_directory = Path(
            output_directory
        )

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save(
        self,
        history,
        filename: str = "training_history.csv",
    ) -> None:

        dataframe = pd.DataFrame(history)

        dataframe.to_csv(

            self.output_directory / filename,

            index=False,

        )