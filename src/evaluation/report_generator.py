"""
Module
-------
Classification Report

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
Produces a detailed per-class performance report.
"""

from __future__ import annotations

from sklearn.metrics import classification_report


class ReportGenerator:
    """
    Creates classification reports.
    """

    @staticmethod
    def create(
        targets,
        predictions,
        class_names,
    ):

        return classification_report(

            targets,

            predictions,

            target_names=class_names,

            digits=4,

            zero_division=0,

            output_dict=True,

        )