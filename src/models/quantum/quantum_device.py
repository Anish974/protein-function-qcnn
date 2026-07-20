"""
Module
-------
Quantum Device

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
Creates and manages the PennyLane simulation backend.

Usage
-----
The project uses analytic expectation values by setting shots=None,
ensuring deterministic gradients during optimization.
"""

from __future__ import annotations

import pennylane as qml


class QuantumDevice:
    """
    Creates PennyLane simulation devices.
    """

    def __init__(
        self,
        number_of_qubits: int = 10,
    ) -> None:

        self.number_of_qubits = number_of_qubits

    def build(self):
        """
        Returns
        -------
        PennyLane Device
            Default statevector simulator.
        """

        return qml.device(
            "default.qubit",
            wires=self.number_of_qubits,
            shots=None,
        )