"""
Module
-------
Ontology Validator

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
Provides comprehensive validation utilities for ontology-based
protein function prediction. The validator performs structural,
numerical and semantic validation before evaluation metrics are
computed.

The validator is intentionally independent from the evaluation
pipeline so that it can be reused by training, validation,
cross-validation and inference workflows.

Usage
-----
>>> validator = OntologyValidator(configuration)
>>> validator.validate(
...     y_true,
...     y_probability,
...     go_terms
... )
"""

from __future__ import annotations

import logging
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence

import numpy as np

from .ontology_configuration import EvaluationConfiguration
from .ontology_utils import (
    contains_inf,
    contains_nan,
    duplicated_labels,
    validate_probability_range,
)

LOGGER = logging.getLogger(__name__)


class OntologyValidationError(ValueError):
    """
    Raised when ontology validation fails.
    """


class OntologyValidator:
    """
    Validates ontology predictions and annotations before
    evaluation.

    The validator checks

    * array dimensions
    * probability ranges
    * NaN values
    * infinite values
    * duplicate GO identifiers
    * threshold consistency
    * ontology consistency
    * multilabel compatibility
    """

    def __init__(
        self,
        configuration: EvaluationConfiguration,
    ) -> None:
        """
        Initialise validator.

        Args:
            configuration:
                Ontology evaluation configuration.
        """

        self.configuration = configuration

    ####################################################################
    # Public API
    ####################################################################

    def validate(
        self,
        y_true: np.ndarray,
        y_probability: np.ndarray,
        go_terms: Optional[Sequence[str]] = None,
    ) -> None:
        """
        Execute all validation routines.

        Args:
            y_true:
                Binary ontology labels.

            y_probability:
                Prediction probabilities.

            go_terms:
                Ordered ontology identifiers.

        Raises:
            OntologyValidationError:
                If validation fails.
        """

        LOGGER.info("Starting ontology validation.")

        self.validate_shapes(
            y_true,
            y_probability,
        )

        self.validate_numeric_content(
            y_true,
            y_probability,
        )

        self.validate_probability_values(
            y_probability,
        )

        self.validate_threshold_configuration()

        if go_terms is not None:
            self.validate_go_terms(go_terms)

        LOGGER.info("Ontology validation completed successfully.")

    ####################################################################
    # Shape validation
    ####################################################################

    def validate_shapes(
        self,
        y_true: np.ndarray,
        y_probability: np.ndarray,
    ) -> None:
        """
        Validate matrix dimensions.

        Args:
            y_true:
                Binary labels.

            y_probability:
                Prediction probabilities.

        Raises:
            OntologyValidationError:
                Shape mismatch.
        """

        if y_true.ndim != 2:

            raise OntologyValidationError(
                "Ground truth array must be two-dimensional."
            )

        if y_probability.ndim != 2:

            raise OntologyValidationError(
                "Prediction array must be two-dimensional."
            )

        if y_true.shape != y_probability.shape:

            raise OntologyValidationError(
                "Prediction and label matrices must have identical shapes."
            )

        LOGGER.debug(
            "Validated matrix dimensions: %s",
            y_true.shape,
        )

    ####################################################################
    # Numerical validation
    ####################################################################

    def validate_numeric_content(
        self,
        y_true: np.ndarray,
        y_probability: np.ndarray,
    ) -> None:
        """
        Detect invalid numeric values.

        Args:
            y_true:
                Label matrix.

            y_probability:
                Prediction matrix.
        """

        if contains_nan(y_true):

            raise OntologyValidationError(
                "Ground truth contains NaN values."
            )

        if contains_nan(y_probability):

            raise OntologyValidationError(
                "Prediction matrix contains NaN values."
            )

        if contains_inf(y_true):

            raise OntologyValidationError(
                "Ground truth contains infinite values."
            )

        if contains_inf(y_probability):

            raise OntologyValidationError(
                "Prediction matrix contains infinite values."
            )

    ####################################################################
    # Probability validation
    ####################################################################

    def validate_probability_values(
        self,
        probabilities: np.ndarray,
    ) -> None:
        """
        Validate probability range.

        Args:
            probabilities:
                Prediction probabilities.
        """

        if not validate_probability_range(probabilities):

            raise OntologyValidationError(
                "Prediction probabilities must lie in [0,1]."
            )

    ####################################################################
    # Threshold validation
    ####################################################################

    def validate_threshold_configuration(
        self,
    ) -> None:
        """
        Validate threshold optimisation settings.
        """

        configuration = self.configuration.threshold

        if configuration.minimum < 0.0:

            raise OntologyValidationError(
                "Minimum threshold must be non-negative."
            )

        if configuration.maximum > 1.0:

            raise OntologyValidationError(
                "Maximum threshold cannot exceed one."
            )

        if configuration.minimum >= configuration.maximum:

            raise OntologyValidationError(
                "Invalid threshold search interval."
            )

        if configuration.step <= 0:

            raise OntologyValidationError(
                "Threshold search step must be positive."
            )

    ####################################################################
    # GO term validation
    ####################################################################

    def validate_go_terms(
        self,
        go_terms: Sequence[str],
    ) -> None:
        """
        Validate ontology identifiers.

        Args:
            go_terms:
                Ordered GO identifiers.
        """

        if len(go_terms) == 0:

            raise OntologyValidationError(
                "Ontology contains no GO terms."
            )

        duplicates = duplicated_labels(go_terms)

        if duplicates:

            raise OntologyValidationError(
                f"Duplicate GO identifiers detected: "
                f"{', '.join(duplicates[:10])}"
            )

        LOGGER.debug(
            "Validated %d GO identifiers.",
            len(go_terms),
        )
    ####################################################################
    # Binary label validation
    ####################################################################

    def validate_binary_labels(
        self,
        y_true: np.ndarray,
    ) -> None:
        """
        Validate that ground-truth labels are binary.

        Args:
            y_true:
                Binary ontology label matrix.

        Raises:
            OntologyValidationError:
                If non-binary values are detected.
        """

        unique_values = np.unique(y_true)

        valid_values = {0, 1}

        if not set(unique_values).issubset(valid_values):

            raise OntologyValidationError(
                "Ground-truth labels must contain only binary "
                f"values. Detected values: {unique_values.tolist()}"
            )

        LOGGER.debug(
            "Binary label validation completed successfully."
        )

    ####################################################################
    # Empty class validation
    ####################################################################

    def validate_empty_classes(
        self,
        y_true: np.ndarray,
        go_terms: Optional[Sequence[str]] = None,
    ) -> List[int]:
        """
        Identify ontology classes without positive annotations.

        Args:
            y_true:
                Binary annotation matrix.

            go_terms:
                Ordered GO identifiers.

        Returns:
            List of empty class indices.
        """

        positives = np.sum(y_true, axis=0)

        empty_classes = np.where(positives == 0)[0].tolist()

        if empty_classes:

            LOGGER.warning(
                "%d ontology classes contain no positive samples.",
                len(empty_classes),
            )

            if go_terms is not None:

                names = [
                    go_terms[index]
                    for index in empty_classes[:10]
                ]

                LOGGER.warning(
                    "Example empty GO terms: %s",
                    ", ".join(names),
                )

        return empty_classes

    ####################################################################
    # Empty sample validation
    ####################################################################

    def validate_empty_samples(
        self,
        y_true: np.ndarray,
    ) -> List[int]:
        """
        Detect proteins without ontology annotations.

        Args:
            y_true:
                Annotation matrix.

        Returns:
            Indices of empty samples.
        """

        annotation_count = np.sum(y_true, axis=1)

        empty_samples = np.where(annotation_count == 0)[0].tolist()

        if empty_samples:

            LOGGER.warning(
                "%d samples contain no ontology annotations.",
                len(empty_samples),
            )

        return empty_samples

    ####################################################################
    # Multilabel validation
    ####################################################################

    def validate_multilabel_structure(
        self,
        y_true: np.ndarray,
    ) -> None:
        """
        Validate multilabel annotation structure.

        Args:
            y_true:
                Annotation matrix.

        Raises:
            OntologyValidationError:
                Invalid multilabel configuration.
        """

        labels_per_sample = np.sum(y_true, axis=1)

        if np.any(labels_per_sample < 0):

            raise OntologyValidationError(
                "Negative annotation count detected."
            )

        if np.max(labels_per_sample) == 0:

            raise OntologyValidationError(
                "Dataset contains no ontology annotations."
            )

        LOGGER.debug(
            "Average ontology labels per sample: %.2f",
            float(np.mean(labels_per_sample)),
        )

    ####################################################################
    # Ontology consistency
    ####################################################################

    def validate_ontology_consistency(
        self,
        y_true: np.ndarray,
        go_terms: Sequence[str],
    ) -> None:
        """
        Validate consistency between annotation matrix and
        ontology identifiers.

        Args:
            y_true:
                Annotation matrix.

            go_terms:
                GO identifiers.

        Raises:
            OntologyValidationError:
                Inconsistent ontology.
        """

        if y_true.shape[1] != len(go_terms):

            raise OntologyValidationError(
                "Number of ontology columns does not match "
                "number of GO identifiers."
            )

    ####################################################################
    # Summary report
    ####################################################################

    def validation_report(
        self,
        y_true: np.ndarray,
        y_probability: np.ndarray,
        go_terms: Optional[Sequence[str]] = None,
    ) -> Dict[str, float]:
        """
        Produce a validation summary.

        Args:
            y_true:
                Binary labels.

            y_probability:
                Prediction probabilities.

            go_terms:
                GO identifiers.

        Returns:
            Validation summary dictionary.
        """

        empty_classes = self.validate_empty_classes(
            y_true,
            go_terms,
        )

        empty_samples = self.validate_empty_samples(
            y_true,
        )

        report = {
            "samples": int(y_true.shape[0]),
            "ontology_terms": int(y_true.shape[1]),
            "empty_classes": len(empty_classes),
            "empty_samples": len(empty_samples),
            "mean_probability": float(
                np.mean(y_probability)
            ),
            "max_probability": float(
                np.max(y_probability)
            ),
            "min_probability": float(
                np.min(y_probability)
            ),
            "mean_labels_per_sample": float(
                np.mean(np.sum(y_true, axis=1))
            ),
        }

        return report

    ####################################################################
    # Complete validation pipeline
    ####################################################################

    def full_validation(
        self,
        y_true: np.ndarray,
        y_probability: np.ndarray,
        go_terms: Sequence[str],
    ) -> Dict[str, float]:
        """
        Execute every validation routine.

        Args:
            y_true:
                Binary labels.

            y_probability:
                Prediction probabilities.

            go_terms:
                Ordered GO identifiers.

        Returns:
            Validation report.
        """

        LOGGER.info(
            "Executing complete ontology validation pipeline."
        )

        self.validate(
            y_true,
            y_probability,
            go_terms,
        )

        self.validate_binary_labels(
            y_true,
        )

        self.validate_multilabel_structure(
            y_true,
        )

        self.validate_ontology_consistency(
            y_true,
            go_terms,
        )

        report = self.validation_report(
            y_true,
            y_probability,
            go_terms,
        )

        LOGGER.info(
            "Ontology validation completed successfully."
        )

        return report