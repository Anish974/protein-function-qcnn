"""
Module
-------
ROC Curve Plot

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
Produces quality ROC figures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


class ROCPlot:
    """
    Draw ROC curves.
    """

    @staticmethod
    def save(

        roc_results,

        class_names,

        output_directory,

        figure_name="Receiver_Operating_Characteristic_Per_Class",

    ):

        output_directory = Path(output_directory)

        output_directory.mkdir(

            parents=True,

            exist_ok=True,

        )

        plt.figure(

            figsize=(8, 8),

            dpi=300,

        )

        for class_index, class_name in enumerate(class_names):

            plt.plot(

                roc_results["fpr"][class_index],

                roc_results["tpr"][class_index],

                linewidth=2,

                label=(
                    f"{class_name} "
                    f"(AUC={roc_results['auc'][class_index]:.4f})"
                ),

            )

        plt.plot(

            [0, 1],

            [0, 1],

            linestyle="--",

            linewidth=1.5,

        )

        plt.xlabel("False Positive Rate")

        plt.ylabel("True Positive Rate")

        plt.title(

            "Receiver Operating Characteristics"

        )

        plt.legend(

            fontsize=8,

            loc="lower right",

        )

        plt.tight_layout()

        for extension in (

            ".png",

            ".pdf",

            ".svg",

        ):

            plt.savefig(

                output_directory /

                f"{figure_name}{extension}",

                bbox_inches="tight",

            )

        plt.close()