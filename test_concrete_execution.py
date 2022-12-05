from quavl.lib.constants import SWAP_matrix, CNOT_matrix
from quavl.lib.expressions.qbit import Qbits

import numpy as np

from quavl.lib.models.circuit import Circuit
from quavl.lib.operations.gates import CNOT, H, SWAP, CZ
from quavl.lib.operations.measurements import measure

delta = 0.0001


def execute_cnot():
    # Initialize circuit
    (a, b) = Qbits(['a', 'b'])

    circuit = Circuit([a, b],
                      [
                          CNOT(a, b)
                      ])

    # Concrete execution
    a_value = (0, 1)
    b_value = (1, 0)

    circuit.initialize([a_value, b_value])
    output = circuit.execute()

    specified_output = np.matmul(CNOT_matrix, np.kron(np.array([a_value]).T, np.array([b_value]).T))

    assert np.allclose(output, specified_output, delta)


def execute_circuit_swapping():
    (a, b) = Qbits(['a', 'b'])

    circuit = Circuit([a, b],
                      [
                          CNOT(a, b),
                          CNOT(b, a),
                          CNOT(a, b)
                      ])

    # Concrete execution
    circuit.initialize([(0, 1), (1, 0)])
    output = circuit.execute()

    assert np.allclose(output, np.matmul(SWAP_matrix, np.kron(np.array([[0, 1]]).T, np.array([[1, 0]]).T)), delta)


def execute_quantum_teleportation():
    # Initialize circuit
    (psi, b0, b1) = Qbits(['psi', 'b0', 'b1'])

    circuit = Circuit([psi, b0, b1],
                      [
                          CNOT(psi, b0),
                          H(psi),
                          CNOT(b0, b1),
                          SWAP(psi, b0),
                          CZ(b0, b1),
                          SWAP(psi, b0),
                          measure(psi),
                          measure(b0)
                      ])

    # Concrete execution
    circuit.initialize([(1 / np.sqrt(2), 1 / np.sqrt(2)), (1, 0), (1, 0)])

    circuit.set_initial_gate_applications([
        H(b0),
        CNOT(b0, b1)
    ])

    circuit.execute()


if __name__ == "__main__":
    execute_circuit_swapping()
    execute_quantum_teleportation()
