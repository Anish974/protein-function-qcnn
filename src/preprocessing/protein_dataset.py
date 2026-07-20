"""
Module
------
Protein Dataset

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
PyTorch dataset implementation for hybrid quantum-classical learning.
"""

from typing import List

import torch

from torch.utils.data import Dataset


class ProteinDataset(Dataset):
    """
    Protein dataset.
    """

    def __init__(

        self,

        sequences: List[List[int]],

        labels: List[int],

    ):

        self.sequences = sequences

        self.labels = labels

    def __len__(self) -> int:

        return len(self.sequences)

    def __getitem__(

        self,

        index: int,

    ):

        sequence = torch.tensor(

            self.sequences[index],

            dtype=torch.long,

        )

        label = torch.tensor(

            self.labels[index],

            dtype=torch.long,

        )

        return sequence, label