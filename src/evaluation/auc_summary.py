"""
Module
-------
AUC Summary

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
Summarizes AUC values into a publication-ready table.
"""

from __future__ import annotations

import pandas as pd


class AUCSummary:
    """
    Creates AUC summary tables.
    """

    @staticmethod
    def create(

        auc_dictionary,

        class_names,

    ) -> pd.DataFrame:

        rows = []

        for index, class_name in enumerate(class_names):

            rows.append(

                {

                    "Protein Class": class_name,

                    "ROC AUC": auc_dictionary[index],

                }

            )

        dataframe = pd.DataFrame(rows)

        dataframe.loc[len(dataframe)] = {

            "Protein Class": "Macro Average",

            "ROC AUC": dataframe["ROC AUC"].mean(),

        }

        return dataframe