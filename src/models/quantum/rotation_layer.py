"""
Module
-------
Parameterized Rotation Layer

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
Applies trainable rotations on every qubit.
"""

from __future__ import annotations

import pennylane as qml


class RotationLayer:
    """
    RX-RY-RZ rotation block.
    """

    @staticmethod
    def apply(
        parameters,
        number_of_qubits: int,
    ) -> None:
        """
        Parameters
        ----------
        parameters
            Shape:
            (qubits,3)
        """

        for wire in range(number_of_qubits):

            qml.RX(
                parameters[wire][0],
                wires=wire,
            )

            qml.RY(
                parameters[wire][1],
                wires=wire,
            )

            qml.RZ(
                parameters[wire][2],
                wires=wire,
            )