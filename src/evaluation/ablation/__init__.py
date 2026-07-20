"""
Module
-------
Ablation Framework

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
Provides the infrastructure required to execute,
evaluate and compare architectural ablation studies
for the Hybrid Quantum-Classical Protein Sequence
Classification framework.
"""

from .experiment import AblationExperiment
from .runner import AblationRunner
from .registry import AblationRegistry
from .results import AblationResults