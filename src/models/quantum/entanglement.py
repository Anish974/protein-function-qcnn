"""
Module
-------
Ring Entanglement

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
Introduces correlations between neighbouring qubits
using a ring topology.
"""

from __future__ import annotations

import pennylane as qml


class RingEntanglement:
    """
    Circular CNOT entanglement.
    """

    @staticmethod
    def apply(
        number_of_qubits: int,
    ) -> None:
        """
        Creates

        0 → 1
        1 → 2
        ...
        n-1 → 0
        """

        for wire in range(number_of_qubits - 1):

            qml.CNOT(
                wires=[
                    wire,
                    wire + 1,
                ]
            )

        qml.CNOT(
            wires=[
                number_of_qubits - 1,
                0,
            ]
        )