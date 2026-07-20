"""
Module
------
Dataset Splitter

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
Creates reproducible stratified train-validation partitions.
"""

from typing import List
from sklearn.model_selection import train_test_split


class DatasetSplitter:
    """
    Stratified dataset partitioning.
    """

    def split(

        self,

        sequences: List,

        labels: List[int],

        validation_size: float = 0.20,

        random_seed: int = 42,

    ):

        return train_test_split(

            sequences,

            labels,

            test_size=validation_size,

            random_state=random_seed,

            stratify=labels,

        )