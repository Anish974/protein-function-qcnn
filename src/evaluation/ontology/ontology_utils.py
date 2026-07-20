"""
Module
-------
Ontology Utility Functions

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
Provides reusable utility functions shared by the ontology
evaluation framework. The utilities cover directory handling,
logging, reproducibility, file serialization, probability
processing, metric formatting, and common validation helpers.

Usage
-----
This module is intended to be imported by all ontology
evaluation components rather than duplicating common logic.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


LOGGER = logging.getLogger(__name__)


def configure_logger(
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure the ontology logger.

    Args:
        level:
            Logging level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger("ontology")

    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(message)s"
        )

        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        logger.addHandler(handler)

    logger.setLevel(level)

    return logger


def create_directory(
    directory: Path | str,
) -> Path:
    """
    Create a directory if it does not exist.

    Args:
        directory:
            Directory path.

    Returns:
        Path object.
    """
    path = Path(directory)

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def ensure_parent_directory(
    file_path: Path | str,
) -> None:
    """
    Ensure parent directory exists.

    Args:
        file_path:
            Output file path.
    """
    Path(file_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )


def current_timestamp() -> str:
    """
    Returns timestamp string.

    Returns:
        Current timestamp.
    """
    return datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )


def initialise_seed(
    seed: int = 42,
) -> None:
    """
    Initialise reproducible random states.

    Args:
        seed:
            Random seed.
    """
    random.seed(seed)

    np.random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    LOGGER.info("Random seed initialised.")


def configuration_hash(
    configuration: Dict[str, Any],
) -> str:
    """
    Compute configuration hash.

    Args:
        configuration:
            Configuration dictionary.

    Returns:
        SHA256 digest.
    """
    encoded = json.dumps(
        configuration,
        sort_keys=True,
        default=str,
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def save_json(
    data: Dict[str, Any],
    file_path: Path | str,
) -> None:
    """
    Save JSON file.

    Args:
        data:
            Dictionary.
        file_path:
            Destination.
    """
    ensure_parent_directory(file_path)

    with open(
        file_path,
        "w",
        encoding="utf-8",
    ) as stream:

        json.dump(
            data,
            stream,
            indent=4,
            default=str,
        )


def load_json(
    file_path: Path | str,
) -> Dict[str, Any]:
    """
    Load JSON file.

    Args:
        file_path:
            JSON path.

    Returns:
        Parsed dictionary.
    """
    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as stream:

        return json.load(stream)


def clip_probabilities(
    probabilities: np.ndarray,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """
    Avoid exact zero and one probabilities.

    Args:
        probabilities:
            Probability matrix.
        epsilon:
            Numerical stability constant.

    Returns:
        Clipped probabilities.
    """
    return np.clip(
        probabilities,
        epsilon,
        1.0 - epsilon,
    )


def normalise_probabilities(
    probabilities: np.ndarray,
) -> np.ndarray:
    """
    Normalise row probabilities.

    Args:
        probabilities:
            Probability matrix.

    Returns:
        Normalised probabilities.
    """
    denominator = probabilities.sum(
        axis=1,
        keepdims=True,
    )

    denominator = np.where(
        denominator == 0,
        1.0,
        denominator,
    )

    return probabilities / denominator


def validate_probability_range(
    probabilities: np.ndarray,
) -> bool:
    """
    Validate probability values.

    Args:
        probabilities:
            Prediction probabilities.

    Returns:
        True if valid.
    """
    return bool(
        np.all(probabilities >= 0.0)
        and np.all(probabilities <= 1.0)
    )


def contains_nan(
    array: np.ndarray,
) -> bool:
    """
    Detect NaN values.

    Args:
        array:
            Input array.

    Returns:
        True if NaN exists.
    """
    return bool(np.isnan(array).any())


def contains_inf(
    array: np.ndarray,
) -> bool:
    """
    Detect infinite values.

    Args:
        array:
            Input array.

    Returns:
        True if Inf exists.
    """
    return bool(np.isinf(array).any())


def unique_labels(
    labels: Sequence[str],
) -> List[str]:
    """
    Return sorted unique labels.

    Args:
        labels:
            Label sequence.

    Returns:
        Sorted labels.
    """
    return sorted(set(labels))


def duplicated_labels(
    labels: Sequence[str],
) -> List[str]:
    """
    Identify duplicated labels.

    Args:
        labels:
            Label sequence.

    Returns:
        Duplicate labels.
    """
    seen = set()

    duplicates = set()

    for label in labels:

        if label in seen:
            duplicates.add(label)

        seen.add(label)

    return sorted(duplicates)


def ontology_display_name(
    ontology: str,
) -> str:
    """
    Human-readable ontology name.

    Args:
        ontology:
            MF/BP/CC.

    Returns:
        Display name.
    """
    mapping = {
        "MF": "Molecular Function",
        "BP": "Biological Process",
        "CC": "Cellular Component",
    }

    return mapping.get(
        ontology,
        ontology,
    )


def metric_display_name(
    metric: str,
) -> str:
    """
    Pretty metric title.

    Args:
        metric:
            Internal metric name.

    Returns:
        Display name.
    """
    return (
        metric.replace("_", " ")
        .title()
        .replace("Auc", "AUC")
        .replace("Aupr", "AUPR")
    )


def format_metric(
    value: float,
    digits: int = 4,
) -> str:
    """
    Format metric.

    Args:
        value:
            Metric value.
        digits:
            Decimal places.

    Returns:
        Formatted string.
    """
    return f"{value:.{digits}f}"


def average_metrics(
    metrics: Iterable[float],
) -> float:
    """
    Average metric values.

    Args:
        metrics:
            Metric iterable.

    Returns:
        Mean value.
    """
    values = list(metrics)

    if not values:
        return 0.0

    return float(np.mean(values))


def standard_deviation(
    metrics: Iterable[float],
) -> float:
    """
    Compute standard deviation.

    Args:
        metrics:
            Metric iterable.

    Returns:
        Standard deviation.
    """
    values = list(metrics)

    if len(values) < 2:
        return 0.0

    return float(np.std(values))


def confidence_interval(
    values: Sequence[float],
    confidence: float = 0.95,
) -> tuple[float, float]:
    """
    Approximate confidence interval.

    Args:
        values:
            Sample values.
        confidence:
            Confidence level.

    Returns:
        Lower and upper bounds.
    """
    data = np.asarray(values)

    mean = np.mean(data)

    std = np.std(data)

    margin = 1.96 * std / np.sqrt(len(data))

    return (
        float(mean - margin),
        float(mean + margin),
    )


def results_directory(
    root: Path,
    ontology: str,
) -> Path:
    """
    Create ontology-specific result folder.

    Args:
        root:
            Base directory.
        ontology:
            Ontology code.

    Returns:
        Result directory.
    """
    path = root / ontology

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path