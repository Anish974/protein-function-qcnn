"""
Module
------
Class Weight Estimation

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
Computes inverse-frequency class weights for weighted cross-entropy.
"""

from typing import List

import numpy as np

from sklearn.utils.class_weight import compute_class_weight


class ClassWeightCalculator:
    """
    Computes balanced class weights.
    """

    def compute(

        self,

        labels: List[int],

    ):

        classes = np.unique(labels)

        weights = compute_class_weight(

            class_weight="balanced",

            classes=classes,

            y=labels,

        )

        return weights