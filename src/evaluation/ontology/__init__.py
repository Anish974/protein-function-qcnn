"""
Module
-------
Ontology Evaluation Package

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
Provides a unified interface for ontology-wise evaluation of
protein function prediction models. The package contains
configuration management, validation, metric computation,
threshold optimization, statistical analysis, visualization,
export utilities, and publication report generation for the
Gene Ontology domains:

    • Molecular Function (MF)
    • Biological Process (BP)
    • Cellular Component (CC)

The package is designed to integrate with the hybrid
CNN–Quantum Variational Classifier framework while remaining
generic enough to support other protein function prediction
models.
"""

from .ontology_configuration import (
    EvaluationConfiguration,
    OntologyConfiguration,
    OntologyNames,
    MetricNames,
)

from .ontology_validator import OntologyValidator

from .ontology_metrics import OntologyMetrics

from .threshold_optimizer import ThresholdOptimizer

from .ontology_evaluator import OntologyEvaluator

from .ontology_summary import OntologySummary

from .ontology_statistics import OntologyStatistics

from .ontology_visualization import OntologyVisualization

from .ontology_export import OntologyExporter

from .ontology_report import OntologyReport

__all__ = [
    "EvaluationConfiguration",
    "OntologyConfiguration",
    "OntologyNames",
    "MetricNames",
    "OntologyValidator",
    "OntologyMetrics",
    "ThresholdOptimizer",
    "OntologyEvaluator",
    "OntologySummary",
    "OntologyStatistics",
    "OntologyVisualization",
    "OntologyExporter",
    "OntologyReport",
]

__author__ = "Amit Pimpalkar"
__organization__ = "RBU, Nagpur"
__year__ = "2026"
__version__ = "1.0.0"