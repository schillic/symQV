import numpy as np

from symqv.lib.constants import CCX_matrix, CNOT_matrix, I_matrix
from symqv.lib.expressions.qbit import Qbits
from symqv.lib.models.circuit import Circuit
from symqv.lib.operations.gates import CNOT, CCX
from symqv.lib.utils.arithmetic import kron
from symqv.lib.solver import SpecificationType


def prove_half_adder():
    # Initialize circuit
    (a, b, z) = Qbits(['a', 'b', 'z'])

    circuit = Circuit([a, b, z],
                      [
                          CCX(a, b, z),
                          CNOT(a, b)
                      ])

    circuit.initialize([{(1, 0), (0, 1)}, {(1, 0), (0, 1)}, (1, 0)])

    # Symbolic execution
    (a_final, s_final, c_final, _, _, _) = circuit.get_final_qbits()

    circuit.set_specification(np.matmul(CCX_matrix,
                                        kron([CNOT_matrix, I_matrix])),
                              SpecificationType.transformation_matrix)

    circuit.prove()


if __name__ == "__main__":
    prove_half_adder()
