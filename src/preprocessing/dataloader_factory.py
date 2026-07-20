"""
Module
------
DataLoader Factory

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
Creates training and validation data loaders from ProteinDataset instances.
"""

from torch.utils.data import DataLoader


class DataLoaderFactory:
    """
    Factory for constructing PyTorch DataLoaders.
    """

    @staticmethod
    def create(
        dataset,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
    ) -> DataLoader:
        """
        Build a configured DataLoader.

        Args:
            dataset:
                Dataset instance.

            batch_size:
                Number of samples per batch.

            shuffle:
                Shuffle dataset each epoch.

            num_workers:
                Number of worker processes.

        Returns:
            Configured DataLoader.
        """

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )