"""
Module
-------
Quantum Measurement

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
Measurement utilities used by the Variational Quantum
Classifier.

Each qubit is measured in the Pauli-Z basis and the
expectation values become the quantum feature vector.
"""

from __future__ import annotations

from typing import List

import pennylane as qml


class MeasurementLayer:
    """
    Quantum measurement operations.
    """

    @staticmethod
    def expectation_values(
        number_of_qubits: int,
    ) -> List:
        """
        Returns expectation values for every qubit.

        Args:
            number_of_qubits:
                Total qubits.

        Returns
        -------
        List of expectation operators.
        """

        return [

            qml.expval(
                qml.PauliZ(wire)
            )

            for wire in range(number_of_qubits)

        ]