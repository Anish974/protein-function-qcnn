"""
Module
-------
Confidence Interval

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

from typing import Dict

import numpy as np


class ConfidenceInterval:
    """
    Classical confidence interval computation.
    """

    @staticmethod
    def normal_interval(
        values: np.ndarray,
        z_score: float = 1.96,
    ) -> Dict[str, float]:

        values = np.asarray(values)

        mean = float(values.mean())

        standard_error = float(

            values.std(ddof=1)

            / np.sqrt(len(values))

        )

        margin = z_score * standard_error

        return {

            "mean": mean,

            "lower": mean - margin,

            "upper": mean + margin,

        }