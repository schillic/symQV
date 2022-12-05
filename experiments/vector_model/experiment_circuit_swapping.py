from quavl.lib.expressions.qbit import Qbits
from quavl.lib.models.circuit import Circuit
from quavl.lib.operations.gates import CNOT, SWAP_matrix
from quavl.lib.solver import SpecificationType


# Circuit swapping of two qbits
def prove_circuit_swapping():
    # Initialize circuit
    (a, b) = Qbits(['a', 'b'])

    circuit = Circuit([a, b],
                      [
                          CNOT(a, b),
                          CNOT(b, a),
                          CNOT(a, b)
                      ])

    # Symbolic execution
    circuit.set_specification(SWAP_matrix, SpecificationType.transformation_matrix)

    circuit.prove()


if __name__ == "__main__":
    prove_circuit_swapping()
