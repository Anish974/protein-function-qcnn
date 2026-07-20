"""
Module
-------
Directory Manager

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
Maintains project directory references.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = ROOT / "datasets"

CHECKPOINT_DIR = ROOT / "checkpoints"

LOG_DIR = ROOT / "logs"

EXPERIMENT_DIR = ROOT / "experiments"

CONFIG_DIR = ROOT / "configs"


def initialize_directories() -> None:
    """
    Creates project directories if absent.
    """

    directories = [

        DATASET_DIR,

        CHECKPOINT_DIR,

        LOG_DIR,

        EXPERIMENT_DIR,

    ]

    for directory in directories:

        directory.mkdir(parents=True, exist_ok=True)