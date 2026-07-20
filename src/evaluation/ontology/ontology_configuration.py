"""
Module
-------
Ontology Evaluation Configuration

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
Defines strongly typed configuration objects for ontology-wise
evaluation of protein function prediction models.

The configuration covers

- Ontology definitions
- Metric selection
- Threshold optimization
- Bootstrap settings
- Export options
- Visualization settings
- Validation rules

The module follows a configuration-driven design to avoid
hard-coded evaluation parameters throughout the framework.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

LOGGER = logging.getLogger(__name__)


class OntologyNames(str, Enum):
    """
    Supported Gene Ontology namespaces.
    """

    MOLECULAR_FUNCTION = "MF"
    BIOLOGICAL_PROCESS = "BP"
    CELLULAR_COMPONENT = "CC"


class MetricNames(str, Enum):
    """
    Supported ontology evaluation metrics.
    """

    FMAX = "Fmax"

    PRECISION_MACRO = "PrecisionMacro"
    PRECISION_MICRO = "PrecisionMicro"

    RECALL_MACRO = "RecallMacro"
    RECALL_MICRO = "RecallMicro"

    F1_MACRO = "F1Macro"
    F1_MICRO = "F1Micro"
    F1_WEIGHTED = "F1Weighted"

    ROC_AUC_MACRO = "MacroAUC"
    ROC_AUC_MICRO = "MicroAUC"

    AUPR_MACRO = "MacroAUPR"
    AUPR_MICRO = "MicroAUPR"

    HAMMING = "HammingLoss"

    JACCARD = "Jaccard"

    COVERAGE = "CoverageError"

    LRAP = "LRAP"

    EXACT_MATCH = "ExactMatch"


@dataclass(slots=True)
class ThresholdConfiguration:
    """
    Threshold search configuration.
    """

    enabled: bool = True

    minimum: float = 0.01

    maximum: float = 0.99

    step: float = 0.01


@dataclass(slots=True)
class BootstrapConfiguration:
    """
    Bootstrap validation settings.
    """

    enabled: bool = True

    samples: int = 1000

    confidence_level: float = 0.95

    random_seed: int = 42


@dataclass(slots=True)
class ExportConfiguration:
    """
    Output export configuration.
    """

    csv: bool = True

    excel: bool = True

    json: bool = True

    markdown: bool = True

    latex: bool = True

    overwrite: bool = True


@dataclass(slots=True)
class VisualizationConfiguration:
    """
    Figure export configuration.
    """

    dpi: int = 600

    save_png: bool = True

    save_pdf: bool = True

    save_svg: bool = True

    figure_width: float = 8.0

    figure_height: float = 6.0


@dataclass(slots=True)
class ValidationConfiguration:
    """
    Input validation configuration.
    """

    check_nan: bool = True

    check_inf: bool = True

    check_duplicate_terms: bool = True

    validate_probability_range: bool = True

    validate_shapes: bool = True

    validate_thresholds: bool = True


@dataclass(slots=True)
class OntologyConfiguration:
    """
    Configuration associated with one ontology.
    """

    name: OntologyNames

    threshold: float = 0.50

    optimize_threshold: bool = True

    enabled_metrics: List[MetricNames] = field(
        default_factory=lambda: list(MetricNames)
    )


@dataclass(slots=True)
class EvaluationConfiguration:
    """
    Master ontology evaluation configuration.
    """

    ontologies: List[OntologyConfiguration] = field(
        default_factory=lambda: [
            OntologyConfiguration(
                name=OntologyNames.MOLECULAR_FUNCTION
            ),
            OntologyConfiguration(
                name=OntologyNames.BIOLOGICAL_PROCESS
            ),
            OntologyConfiguration(
                name=OntologyNames.CELLULAR_COMPONENT
            ),
        ]
    )

    threshold: ThresholdConfiguration = field(
        default_factory=ThresholdConfiguration
    )

    bootstrap: BootstrapConfiguration = field(
        default_factory=BootstrapConfiguration
    )

    export: ExportConfiguration = field(
        default_factory=ExportConfiguration
    )

    visualization: VisualizationConfiguration = field(
        default_factory=VisualizationConfiguration
    )

    validation: ValidationConfiguration = field(
        default_factory=ValidationConfiguration
    )

    output_directory: Path = Path("results/ontology")

    random_seed: int = 42

    verbose: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts configuration into a dictionary.

        Returns:
            Dictionary representation.
        """
        return asdict(self)

    def save_json(
        self,
        file_path: Path,
    ) -> None:
        """
        Saves configuration as JSON.

        Args:
            file_path:
                Destination file.
        """

        try:

            file_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with file_path.open(
                "w",
                encoding="utf-8",
            ) as stream:

                json.dump(
                    self.to_dict(),
                    stream,
                    indent=4,
                    default=str,
                )

            LOGGER.info(
                "Configuration written to %s",
                file_path,
            )

        except OSError as error:

            LOGGER.exception(error)

            raise

    @classmethod
    def load_json(
        cls,
        file_path: Path,
    ) -> "EvaluationConfiguration":
        """
        Loads configuration from JSON.

        Args:
            file_path:
                JSON configuration file.

        Returns:
            Evaluation configuration instance.
        """

        if not file_path.exists():

            raise FileNotFoundError(file_path)

        with file_path.open(
            "r",
            encoding="utf-8",
        ) as stream:

            data = json.load(stream)

        configuration = cls()

        if "random_seed" in data:
            configuration.random_seed = data["random_seed"]

        if "verbose" in data:
            configuration.verbose = data["verbose"]

        if "output_directory" in data:
            configuration.output_directory = Path(
                data["output_directory"]
            )

        return configuration

    def ontology(
        self,
        ontology: OntologyNames,
    ) -> Optional[OntologyConfiguration]:
        """
        Retrieves one ontology configuration.

        Args:
            ontology:
                Ontology identifier.

        Returns:
            Matching configuration if available.
        """

        for item in self.ontologies:

            if item.name == ontology:

                return item

        return None

    def validate(self) -> None:
        """
        Validates configuration consistency.

        Raises:
            ValueError:
                Invalid configuration.
        """

        if self.threshold.minimum >= self.threshold.maximum:

            raise ValueError(
                "Minimum threshold must be smaller than maximum threshold."
            )

        if self.threshold.step <= 0:

            raise ValueError(
                "Threshold step must be positive."
            )

        if not (0.0 < self.bootstrap.confidence_level < 1.0):

            raise ValueError(
                "Confidence level must be between 0 and 1."
            )

        if self.bootstrap.samples <= 0:

            raise ValueError(
                "Bootstrap samples must be greater than zero."
            )

        LOGGER.info(
            "Ontology evaluation configuration validated."
        )