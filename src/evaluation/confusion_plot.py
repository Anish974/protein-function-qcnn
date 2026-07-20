"""
Module
-------
Confusion Matrix Visualization

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
Creates quality confusion matrix figures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class ConfusionPlot:
    """
    Draws confusion matrix.
    """

    @staticmethod
    def save(

        matrix: np.ndarray,

        class_names,

        output_path: str,

        title: str,

    ) -> None:

        figure = plt.figure(

            figsize=(10, 8),

            dpi=300,

        )

        axis = figure.add_subplot(111)

        image = axis.imshow(

            matrix,

            interpolation="nearest",

        )

        figure.colorbar(image)

        axis.set_xticks(

            range(len(class_names))

        )

        axis.set_yticks(

            range(len(class_names))

        )

        axis.set_xticklabels(

            class_names,

            rotation=45,

            ha="right",

        )

        axis.set_yticklabels(

            class_names,

        )

        axis.set_xlabel("Predicted Class")

        axis.set_ylabel("True Class")

        axis.set_title(title)

        output = Path(output_path)

        output.parent.mkdir(

            parents=True,

            exist_ok=True,

        )

        figure.tight_layout()

        figure.savefig(

            output,

            bbox_inches="tight",

        )

        plt.close(figure)