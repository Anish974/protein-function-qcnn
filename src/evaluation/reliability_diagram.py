"""
Module
-------
Reliability Diagram

Author
------
Amit Pimpalkar

Organization
------------
RBU, Nagpur

Year
----
2026
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class ReliabilityDiagram:
    """
    Creates reliability diagrams.
    """

    @staticmethod
    def save(
        probabilities,
        targets,
        output_path,
        bins: int = 10,
    ):

        confidence = np.max(probabilities, axis=1)

        prediction = np.argmax(probabilities, axis=1)

        correct = prediction == targets

        boundaries = np.linspace(0, 1, bins + 1)

        x_values = []
        y_values = []

        for index in range(bins):

            mask = (

                (confidence > boundaries[index])

                &

                (confidence <= boundaries[index + 1])

            )

            if np.sum(mask) == 0:
                continue

            x_values.append(

                np.mean(confidence[mask])

            )

            y_values.append(

                np.mean(correct[mask])

            )

        plt.figure(

            figsize=(6, 6),

            dpi=300,

        )

        plt.plot(

            [0, 1],

            [0, 1],

            "--",

            linewidth=2,

        )

        plt.plot(

            x_values,

            y_values,

            marker="o",

            linewidth=2,

        )

        plt.xlabel("Confidence")

        plt.ylabel("Observed Accuracy")

        plt.title("Reliability Diagram")

        output = Path(output_path)

        output.parent.mkdir(

            parents=True,

            exist_ok=True,

        )

        plt.tight_layout()

        plt.savefig(output)

        plt.close()