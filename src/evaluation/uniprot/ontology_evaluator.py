"""
Ontology Evaluation

Evaluates MF, BP and CC independently.
"""

from __future__ import annotations

from typing import Dict

from .auc import ROCAUC
from .aupr import AUPR
from .fmax import FMax


class OntologyEvaluator:
    """
    Complete GO evaluation.
    """

    def evaluate(
        self,
        mf_truth,
        mf_score,
        bp_truth,
        bp_score,
        cc_truth,
        cc_score,
    ) -> Dict:

        results = {}

        results["MF"] = {

            **FMax.compute(

                mf_truth,

                mf_score,

            ),

            **AUPR.compute(

                mf_truth,

                mf_score,

            ),

            **ROCAUC.compute(

                mf_truth,

                mf_score,

            ),

        }

        results["BP"] = {

            **FMax.compute(

                bp_truth,

                bp_score,

            ),

            **AUPR.compute(

                bp_truth,

                bp_score,

            ),

            **ROCAUC.compute(

                bp_truth,

                bp_score,

            ),

        }

        results["CC"] = {

            **FMax.compute(

                cc_truth,

                cc_score,

            ),

            **AUPR.compute(

                cc_truth,

                cc_score,

            ),

            **ROCAUC.compute(

                cc_truth,

                cc_score,

            ),

        }

        return results