from quavl.lib.constants import CNOT_matrix
from quavl.lib.expressions.qbit import Qbits
from quavl.lib.models.circuit import Circuit
from quavl.lib.solver import SpecificationType
from quavl.lib.operations.gates import CNOT


# CNOT Circuit

def prove_cnot():
    # Initialize circuit
    (a, b) = Qbits(['a', 'b'])
    delta = 0.0001

    circuit = Circuit([a, b],
                      [CNOT(a, b)],
                      delta)

    # Symbolic execution
    circuit.set_specification(CNOT_matrix, SpecificationType.transformation_matrix)

    circuit.prove()


if __name__ == "__main__":
    prove_cnot()
