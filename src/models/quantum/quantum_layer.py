"""
Module
-------
Torch Quantum Layer

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
Creates a PennyLane TorchLayer that integrates
with the PyTorch computational graph.
"""

from __future__ import annotations

import pennylane as qml

import torch.nn as nn

from .quantum_device import QuantumDevice
from .variational_circuit import VariationalCircuit


class QuantumLayer(nn.Module):
    """
    PyTorch compatible quantum layer.
    """

    def __init__(
        self,
        number_of_qubits: int = 10,
        circuit_depth: int = 5,
    ) -> None:

        super().__init__()

        device = QuantumDevice(
            number_of_qubits
        ).build()

        circuit = VariationalCircuit(
            number_of_qubits,
            circuit_depth,
        )

        qnode = qml.QNode(

            circuit.build(),

            device,

            interface="torch",

            diff_method="parameter-shift",

        )

        weight_shapes = {

            "weights":

            (
                circuit_depth,

                number_of_qubits,

                3,

            )

        }

        self.quantum = qml.qnn.TorchLayer(

            qnode,

            weight_shapes,

        )

    def forward(
        self,
        x,
    ):

        return self.quantum(x)