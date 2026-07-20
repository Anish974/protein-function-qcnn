"""
Module
-------
Variational Quantum Circuit

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
Defines the complete multi-layer quantum circuit.

Architecture
------------
Angle Encoding

↓

Five Variational Blocks

↓

Ring Entanglement

↓

Expectation Measurement
"""

from __future__ import annotations

import pennylane as qml

from .rotation_layer import RotationLayer
from .entanglement import RingEntanglement
from .measurement import MeasurementLayer


class VariationalCircuit:
    """
    Multi-layer variational quantum circuit.
    """

    def __init__(
        self,
        number_of_qubits: int = 10,
        circuit_depth: int = 5,
    ) -> None:

        self.number_of_qubits = number_of_qubits

        self.circuit_depth = circuit_depth

    def build(self):

        def quantum_circuit(
            inputs,
            weights,
        ):

            # Data encoding
            for wire in range(self.number_of_qubits):

                qml.RY(
                    inputs[wire],
                    wires=wire,
                )

            # Data re-uploading with trainable layers
            for layer in range(self.circuit_depth):

                RotationLayer.apply(
                    weights[layer],
                    self.number_of_qubits,
                )

                RingEntanglement.apply(
                    self.number_of_qubits,
                )

                # Re-upload classical information
                for wire in range(self.number_of_qubits):

                    qml.RY(
                        inputs[wire],
                        wires=wire,
                    )

            return MeasurementLayer.expectation_values(
                self.number_of_qubits
            )

        return quantum_circuit