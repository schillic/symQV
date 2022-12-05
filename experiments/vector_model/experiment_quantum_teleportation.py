from quavl.lib.expressions.qbit import Qbits
from quavl.lib.models.circuit import Circuit
from quavl.lib.solver import SpecificationType
from quavl.lib.operations.gates import CNOT, H, CZ, SWAP
from quavl.lib.operations.measurements import measure


# Quantum teleportation
def prove_quantum_teleportation():
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

    circuit.initialize([None, (1, 0), (1, 0)])

    circuit.set_initial_gate_applications([
        H(b0),
        CNOT(b0, b1)
    ])

    # Symbolic execution
    (psi_final, b0_final, b1_final) = circuit.get_final_qbits()

    circuit.set_specification((psi, b1_final), SpecificationType.equality_pair)

    circuit.prove()


if __name__ == "__main__":
    prove_quantum_teleportation()
