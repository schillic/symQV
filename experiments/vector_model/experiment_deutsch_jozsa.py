from typing import List

import numpy as np

from symqv.lib.expressions.qbit import Qbits, QbitVal
from symqv.lib.models.circuit import Circuit
from symqv.lib.operations.gates import H, X, CNOT, I
from symqv.lib.operations.measurements import measure
from symqv.lib.utils.arithmetic import kron


def prove_deutsch_jozsa(n: int = 3):
    """
    Correctness proof of the Deutsch-Jozsa algorithm.
    :param n: number of qbits.
    """
    qbits = Qbits([f'q{i}' for i in range(n)])

    for balanced in [True, False]:
        # Create oracles
        oracles = create_oracles(qbits, balanced)
        oracle = oracles[0]

        # Initialize circuit
        circuit = Circuit(qbits,
                          [
                              [H(q) for q in qbits],
                              oracle,
                              [H(q) for q in qbits],
                              measure(qbits[:-1])
                          ])

        circuit.initialize([(1, 0) for _ in qbits[:-1]] + [(0, 1)])

        # Symbolic execution
        final_qbits = circuit.get_final_qbits()

        spec = kron([np.array([[1], [0]]) for _ in qbits[:-1]] + [np.array([[0], [1]])])
        spec = spec.T[0]

        #if balanced:
            # balanced oracle: we never measure the all-zero state.
            #circuit.set_specification(spec, SpecificationType.final_state_vector, is_equality_specification=False)
        #else:
            # constant oracle: states before and after querying the oracle are the same.
            # circuit.set_specification(spec, SpecificationType.final_state_vector)

        circuit.prove()


def create_oracles(qbits: List[QbitVal], balanced: bool):
    """
    Create all possible balanced or constant oracles.
    :param qbits: qbits in the system.
    :param balanced: true if balanced, false if constant.
    :return: list of oracles.
    """
    if balanced:
        # balanced
        operations = []
        num_digits = len('{0:b}'.format(2 ** len(qbits))) - 1
        binary_format = '{0:0' + str(num_digits) + 'b}'

        for i in range(1, len(qbits) - 1):
            balanced_op = []

            bit_vector = binary_format.format(i)

            for j, b in enumerate(bit_vector[::-1]):
                if b == '1':
                    balanced_op.append(CNOT(qbits[j], qbits[-1]))

            operations.append(balanced_op)
            operations.append(balanced_op + [X(qbits[-1])])

        return operations

    # constant
    return [[I(qbits[-1])], [X(qbits[-1])]]


if __name__ == "__main__":
    prove_deutsch_jozsa(n=3)
