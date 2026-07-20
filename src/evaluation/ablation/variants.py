"""
Module
-------
Ablation Variants

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
Defines all ablation configurations for the Hybrid
Quantum-Classical Protein Sequence Classification framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class AblationVariant:
    """
    Configuration for a single ablation experiment.
    """

    name: str
    cnn: bool = True
    quantum: bool = True
    multiscale: bool = True
    residual: bool = True
    data_reuploading: bool = True
    entanglement: bool = True
    vqc_layers: int = 5


ABLATION_VARIANTS: List[AblationVariant] = [

    AblationVariant(
        name="Proposed HQNN"
    ),

    AblationVariant(
        name="CNN Only",
        quantum=False,
    ),

    AblationVariant(
        name="Quantum Only",
        cnn=False,
    ),

    AblationVariant(
        name="Without Multi-Scale CNN",
        multiscale=False,
    ),

    AblationVariant(
        name="Without Residual Block",
        residual=False,
    ),

    AblationVariant(
        name="Shallow VQC",
        vqc_layers=2,
    ),

    AblationVariant(
        name="Without Data Re-uploading",
        data_reuploading=False,
    ),

    AblationVariant(
        name="Without Entanglement",
        entanglement=False,
    ),
]