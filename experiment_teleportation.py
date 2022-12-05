import time

import numpy as np

from symqv.lib.expressions.qbit import Qbits
from symqv.lib.models.circuit import Circuit, Method
from symqv.lib.operations.gates import CNOT, H, CZ, SWAP
from symqv.lib.operations.measurements import measure
from symqv.lib.solver import SpecificationType


# Quantum teleportation
def prove_quantum_teleportation():
    # Initialize circuit
    (psi, b0, b1) = Qbits(['psi', 'b0', 'b1'])

    circuit = Circuit([psi, b0, b1],
                      [CNOT(psi, b0),
                       H(psi),
                       CNOT(b0, b1),
                       SWAP(psi, b0),
                       CZ(b0, b1),
                       SWAP(psi, b0),
                       measure(psi),
                       measure(b0)])

    circuit.initialize([None, (1, 0), (1, 0)])

    circuit.set_initial_gate_applications([
        H(b0),
        CNOT(b0, b1)
    ])

    # Symbolic execution
    (psi_final, b0_final, b1_final) = circuit.get_final_qbits()

    circuit.set_specification((psi, b1_final), SpecificationType.equality_pair)

    circuit.prove(method=Method.state_model,
                  overapproximation=False)


if __name__ == "__main__":
    times = []

    for _ in range(5):
        start = time.time()
        prove_quantum_teleportation()
        times.append(time.time() - start)

    print(f'Runtime:', np.mean(times))
