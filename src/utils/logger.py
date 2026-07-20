"""
Module
-------
Logging Utility

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
Provides a unified logging interface for all project modules.
"""

import logging

from pathlib import Path


def build_logger(name: str) -> logging.Logger:
    """
    Creates a project logger.

    Args:
        name:
            Logger name.

    Returns:
        Configured logger.
    """

    Path("logs").mkdir(exist_ok=True)

    logger = logging.getLogger(name)

    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(

        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

    )

    file_handler = logging.FileHandler(

        "logs/project.log"

    )

    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()

    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    logger.addHandler(console_handler)

    return logger