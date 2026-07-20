"""
Module
-------
Effect Size

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
Computes practical significance of model differences.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


class EffectSize:
    """
    Effect size statistics.
    """

    @staticmethod
    def cohens_d(
        baseline: np.ndarray,
        proposed: np.ndarray,
    ) -> Dict[str, float]:

        baseline = np.asarray(baseline)

        proposed = np.asarray(proposed)

        pooled = np.sqrt(

            (

                baseline.var(ddof=1)

                +

                proposed.var(ddof=1)

            )

            / 2.0

        )

        d = (

            proposed.mean()

            - baseline.mean()

        ) / pooled

        return {

            "cohens_d": float(d)

        }

    @staticmethod
    def cliffs_delta(
        baseline: np.ndarray,
        proposed: np.ndarray,
    ) -> Dict[str, float]:

        baseline = np.asarray(baseline)

        proposed = np.asarray(proposed)

        greater = 0
        smaller = 0

        for value_a in proposed:

            for value_b in baseline:

                if value_a > value_b:

                    greater += 1

                elif value_a < value_b:

                    smaller += 1

        delta = (

            greater - smaller

        ) / (

            len(proposed)

            * len(baseline)

        )

        return {

            "cliffs_delta": float(delta)

        }