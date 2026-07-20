"""
Module
-------
Ontology Evaluation Metrics

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
Provides reusable ontology evaluation metrics for protein
function prediction. The implementation supports multi-label
classification and ontology-wise evaluation for Molecular
Function (MF), Biological Process (BP), and Cellular
Component (CC).

This module computes threshold-independent metrics and forms
the core numerical engine used by the ontology evaluator.

Usage
-----
>>> metrics = OntologyMetrics(configuration)
>>> results = metrics.compute_basic_metrics(
...     y_true,
...     y_prediction
... )
"""

from __future__ import annotations

import logging
from typing import Dict
from typing import Callable

import numpy as np

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)

from .ontology_configuration import (
    EvaluationConfiguration,
)

LOGGER = logging.getLogger(__name__)


class OntologyMetrics:
    """
    Computes ontology evaluation metrics.

    The class is intentionally stateless so that it can be
    reused during training, validation, cross-validation,
    statistical analysis and inference.
    """

    def __init__(
        self,
        configuration: EvaluationConfiguration,
    ) -> None:
        """
        Initialise metric engine.

        Args:
            configuration:
                Evaluation configuration.
        """

        self.configuration = configuration

        self.metric_registry: Dict[
            str,
            Callable,
        ] = {}

        self._register_default_metrics()

    ####################################################################
    # Metric Registry
    ####################################################################

    def _register_default_metrics(
        self,
    ) -> None:
        """
        Register available metrics.
        """

        self.metric_registry = {

            "precision_macro":
                self.precision_macro,

            "precision_micro":
                self.precision_micro,

            "recall_macro":
                self.recall_macro,

            "recall_micro":
                self.recall_micro,

            "f1_macro":
                self.f1_macro,

            "f1_micro":
                self.f1_micro,

            "f1_weighted":
                self.f1_weighted,
        }

        LOGGER.info(
            "%d ontology metrics registered.",
            len(self.metric_registry),
        )

    ####################################################################
    # Precision
    ####################################################################

    def precision_macro(
        self,
        y_true: np.ndarray,
        y_prediction: np.ndarray,
    ) -> float:
        """
        Compute macro precision.
        """

        return float(
            precision_score(
                y_true,
                y_prediction,
                average="macro",
                zero_division=0,
            )
        )

    def precision_micro(
        self,
        y_true: np.ndarray,
        y_prediction: np.ndarray,
    ) -> float:
        """
        Compute micro precision.
        """

        return float(
            precision_score(
                y_true,
                y_prediction,
                average="micro",
                zero_division=0,
            )
        )

    ####################################################################
    # Recall
    ####################################################################

    def recall_macro(
        self,
        y_true: np.ndarray,
        y_prediction: np.ndarray,
    ) -> float:
        """
        Compute macro recall.
        """

        return float(
            recall_score(
                y_true,
                y_prediction,
                average="macro",
                zero_division=0,
            )
        )

    def recall_micro(
        self,
        y_true: np.ndarray,
        y_prediction: np.ndarray,
    ) -> float:
        """
        Compute micro recall.
        """

        return float(
            recall_score(
                y_true,
                y_prediction,
                average="micro",
                zero_division=0,
            )
        )

    ####################################################################
    # F1 Score
    ####################################################################

    def f1_macro(
        self,
        y_true: np.ndarray,
        y_prediction: np.ndarray,
    ) -> float:
        """
        Compute macro F1.
        """

        return float(
            f1_score(
                y_true,
                y_prediction,
                average="macro",
                zero_division=0,
            )
        )

    def f1_micro(
        self,
        y_true: np.ndarray,
        y_prediction: np.ndarray,
    ) -> float:
        """
        Compute micro F1.
        """

        return float(
            f1_score(
                y_true,
                y_prediction,
                average="micro",
                zero_division=0,
            )
        )

    def f1_weighted(
        self,
        y_true: np.ndarray,
        y_prediction: np.ndarray,
    ) -> float:
        """
        Compute weighted F1.
        """

        return float(
            f1_score(
                y_true,
                y_prediction,
                average="weighted",
                zero_division=0,
            )
        )

    ####################################################################
    # Aggregated Basic Metrics
    ####################################################################

    def compute_basic_metrics(
        self,
        y_true: np.ndarray,
        y_prediction: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute all registered basic metrics.

        Args:
            y_true:
                Binary ontology labels.

            y_prediction:
                Binary predictions.

        Returns:
            Dictionary containing metric values.
        """

        LOGGER.info(
            "Computing ontology classification metrics."
        )

        results: Dict[str, float] = {}

        for metric_name, metric_function in (
            self.metric_registry.items()
        ):

            results[metric_name] = metric_function(
                y_true,
                y_prediction,
            )

        LOGGER.info(
            "Computed %d ontology metrics.",
            len(results),
        )

        return results

    ####################################################################
    # Threshold Utilities
    ####################################################################

    def apply_threshold(
        self,
        probabilities: np.ndarray,
        threshold: float,
    ) -> np.ndarray:
        """
        Convert probability predictions into binary labels.

        Args:
            probabilities:
                Prediction probability matrix.

            threshold:
                Decision threshold.

        Returns:
            Binary prediction matrix.
        """

        return (probabilities >= threshold).astype(np.int32)

    ####################################################################
    # Fmax
    ####################################################################

    def compute_fmax(
        self,
        y_true: np.ndarray,
        y_probability: np.ndarray,
        minimum: float = 0.01,
        maximum: float = 0.99,
        step: float = 0.01,
    ) -> Dict[str, float]:
        """
        Compute the maximum F-score across a threshold sweep.

        Args:
            y_true:
                Binary ontology labels.

            y_probability:
                Prediction probabilities.

            minimum:
                Starting threshold.

            maximum:
                Final threshold.

            step:
                Threshold increment.

        Returns:
            Dictionary containing the optimal threshold,
            Fmax, precision and recall.
        """

        best_f1 = -1.0
        best_precision = 0.0
        best_recall = 0.0
        best_threshold = minimum

        thresholds = np.arange(
            minimum,
            maximum + step,
            step,
        )

        for threshold in thresholds:

            prediction = self.apply_threshold(
                y_probability,
                float(threshold),
            )

            precision = precision_score(
                y_true,
                prediction,
                average="micro",
                zero_division=0,
            )

            recall = recall_score(
                y_true,
                prediction,
                average="micro",
                zero_division=0,
            )

            if precision + recall == 0.0:

                fscore = 0.0

            else:

                fscore = (
                    2.0
                    * precision
                    * recall
                    / (precision + recall)
                )

            if fscore > best_f1:

                best_f1 = fscore
                best_precision = precision
                best_recall = recall
                best_threshold = float(threshold)

        return {
            "fmax": float(best_f1),
            "optimal_threshold": best_threshold,
            "precision": float(best_precision),
            "recall": float(best_recall),
        }

    ####################################################################
    # ROC-AUC
    ####################################################################

    def roc_auc_macro(
        self,
        y_true: np.ndarray,
        y_probability: np.ndarray,
    ) -> float:
        """
        Compute macro ROC-AUC.

        Args:
            y_true:
                Binary ontology labels.

            y_probability:
                Prediction probabilities.

        Returns:
            Macro ROC-AUC.
        """

        from sklearn.metrics import roc_auc_score

        return float(
            roc_auc_score(
                y_true,
                y_probability,
                average="macro",
            )
        )

    def roc_auc_micro(
        self,
        y_true: np.ndarray,
        y_probability: np.ndarray,
    ) -> float:
        """
        Compute micro ROC-AUC.

        Args:
            y_true:
                Binary ontology labels.

            y_probability:
                Prediction probabilities.

        Returns:
            Micro ROC-AUC.
        """

        from sklearn.metrics import roc_auc_score

        return float(
            roc_auc_score(
                y_true,
                y_probability,
                average="micro",
            )
        )

    ####################################################################
    # Precision-Recall Area
    ####################################################################

    def aupr_macro(
        self,
        y_true: np.ndarray,
        y_probability: np.ndarray,
    ) -> float:
        """
        Compute macro AUPR.

        Args:
            y_true:
                Binary labels.

            y_probability:
                Prediction probabilities.

        Returns:
            Macro AUPR.
        """

        from sklearn.metrics import average_precision_score

        return float(
            average_precision_score(
                y_true,
                y_probability,
                average="macro",
            )
        )

    def aupr_micro(
        self,
        y_true: np.ndarray,
        y_probability: np.ndarray,
    ) -> float:
        """
        Compute micro AUPR.

        Args:
            y_true:
                Binary labels.

            y_probability:
                Prediction probabilities.

        Returns:
            Micro AUPR.
        """

        from sklearn.metrics import average_precision_score

        return float(
            average_precision_score(
                y_true,
                y_probability,
                average="micro",
            )
        )

    ####################################################################
    # Probability-Based Evaluation
    ####################################################################

    def compute_probability_metrics(
        self,
        y_true: np.ndarray,
        y_probability: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute threshold-independent ontology metrics.

        Args:
            y_true:
                Binary labels.

            y_probability:
                Prediction probabilities.

        Returns:
            Dictionary of ontology metrics.
        """

        fmax = self.compute_fmax(
            y_true,
            y_probability,
            minimum=self.configuration.threshold.minimum,
            maximum=self.configuration.threshold.maximum,
            step=self.configuration.threshold.step,
        )

        return {
            "fmax": fmax["fmax"],
            "optimal_threshold": fmax["optimal_threshold"],
            "precision_at_fmax": fmax["precision"],
            "recall_at_fmax": fmax["recall"],
            "macro_auc": self.roc_auc_macro(
                y_true,
                y_probability,
            ),
            "micro_auc": self.roc_auc_micro(
                y_true,
                y_probability,
            ),
            "macro_aupr": self.aupr_macro(
                y_true,
                y_probability,
            ),
            "micro_aupr": self.aupr_micro(
                y_true,
                y_probability,
            ),
        }

    ####################################################################
    # Jaccard Similarity
    ####################################################################

    def jaccard_index(
        self,
        y_true: np.ndarray,
        y_prediction: np.ndarray,
    ) -> float:
        """
        Compute the Jaccard similarity coefficient.

        Args:
            y_true:
                Binary ontology labels.

            y_prediction:
                Binary ontology predictions.

        Returns:
            Sample-wise Jaccard similarity.
        """
        from sklearn.metrics import jaccard_score

        return float(
            jaccard_score(
                y_true,
                y_prediction,
                average="samples",
                zero_division=0,
            )
        )

    ####################################################################
    # Hamming Loss
    ####################################################################

    def hamming_loss(
        self,
        y_true: np.ndarray,
        y_prediction: np.ndarray,
    ) -> float:
        """
        Compute multilabel Hamming loss.

        Args:
            y_true:
                Binary ontology labels.

            y_prediction:
                Binary ontology predictions.

        Returns:
            Hamming loss.
        """
        from sklearn.metrics import hamming_loss

        return float(
            hamming_loss(
                y_true,
                y_prediction,
            )
        )

    ####################################################################
    # Coverage Error
    ####################################################################

    def coverage_error(
        self,
        y_true: np.ndarray,
        y_probability: np.ndarray,
    ) -> float:
        """
        Compute coverage error.

        Args:
            y_true:
                Binary ontology labels.

            y_probability:
                Prediction probabilities.

        Returns:
            Coverage error.
        """
        from sklearn.metrics import coverage_error

        return float(
            coverage_error(
                y_true,
                y_probability,
            )
        )

    ####################################################################
    # Label Ranking Average Precision
    ####################################################################

    def label_ranking_average_precision(
        self,
        y_true: np.ndarray,
        y_probability: np.ndarray,
    ) -> float:
        """
        Compute LRAP.

        Args:
            y_true:
                Binary ontology labels.

            y_probability:
                Prediction probabilities.

        Returns:
            Label Ranking Average Precision.
        """
        from sklearn.metrics import (
            label_ranking_average_precision_score,
        )

        return float(
            label_ranking_average_precision_score(
                y_true,
                y_probability,
            )
        )

    ####################################################################
    # Exact Match Ratio
    ####################################################################

    def exact_match_ratio(
        self,
        y_true: np.ndarray,
        y_prediction: np.ndarray,
    ) -> float:
        """
        Compute subset accuracy (exact match ratio).

        Args:
            y_true:
                Binary ontology labels.

            y_prediction:
                Binary ontology predictions.

        Returns:
            Exact match ratio.
        """
        sample_matches = np.all(
            y_true == y_prediction,
            axis=1,
        )

        return float(
            np.mean(sample_matches)
        )

    ####################################################################
    # Per-Class Evaluation
    ####################################################################

    def per_class_metrics(
        self,
        y_true: np.ndarray,
        y_prediction: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Compute precision, recall and F1 for each ontology term.

        Args:
            y_true:
                Binary ontology labels.

            y_prediction:
                Binary ontology predictions.

        Returns:
            Dictionary containing per-class arrays.
        """
        precision = precision_score(
            y_true,
            y_prediction,
            average=None,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            y_prediction,
            average=None,
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            y_prediction,
            average=None,
            zero_division=0,
        )

        support = np.sum(
            y_true,
            axis=0,
        )

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    ####################################################################
    # Confusion Statistics
    ####################################################################

    def multilabel_confusion(
        self,
        y_true: np.ndarray,
        y_prediction: np.ndarray,
    ) -> np.ndarray:
        """
        Compute confusion matrices for each ontology term.

        Args:
            y_true:
                Binary ontology labels.

            y_prediction:
                Binary ontology predictions.

        Returns:
            Multilabel confusion matrix.
        """
        from sklearn.metrics import multilabel_confusion_matrix

        return multilabel_confusion_matrix(
            y_true,
            y_prediction,
        )

    ####################################################################
    # Complete Evaluation API
    ####################################################################

    def evaluate(
        self,
        y_true: np.ndarray,
        y_probability: np.ndarray,
        threshold: float | None = None,
    ) -> Dict[str, object]:
        """
        Compute the complete ontology evaluation.

        Args:
            y_true:
                Binary ontology labels.

            y_probability:
                Prediction probabilities.

            threshold:
                Decision threshold. If omitted, the configured
                threshold is used.

        Returns:
            Dictionary containing all evaluation results.
        """

        if threshold is None:
            threshold = 0.50

        y_prediction = self.apply_threshold(
            y_probability,
            threshold,
        )

        results: Dict[str, object] = {}

        results.update(
            self.compute_basic_metrics(
                y_true,
                y_prediction,
            )
        )

        results.update(
            self.compute_probability_metrics(
                y_true,
                y_probability,
            )
        )

        results["jaccard"] = self.jaccard_index(
            y_true,
            y_prediction,
        )

        results["hamming_loss"] = self.hamming_loss(
            y_true,
            y_prediction,
        )

        results["coverage_error"] = self.coverage_error(
            y_true,
            y_probability,
        )

        results["lrap"] = (
            self.label_ranking_average_precision(
                y_true,
                y_probability,
            )
        )

        results["exact_match"] = (
            self.exact_match_ratio(
                y_true,
                y_prediction,
            )
        )

        results["per_class"] = self.per_class_metrics(
            y_true,
            y_prediction,
        )

        results["confusion_matrix"] = (
            self.multilabel_confusion(
                y_true,
                y_prediction,
            )
        )

        LOGGER.info(
            "Completed ontology evaluation "
            "using %d metrics.",
            len(results),
        )

        return results