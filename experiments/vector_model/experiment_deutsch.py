from typing import Tuple

from symqv.lib.expressions.qbit import Qbits, QbitVal
from symqv.lib.models.circuit import Circuit
from symqv.lib.solver import SpecificationType
from symqv.lib.operations.gates import H, X, CNOT, I
from symqv.lib.operations.measurements import measure


# Deutsch's algorithm
def prove_deutsch_algorithm():
    (q0, q1) = Qbits(['q0', 'q1'])

    cases = {
        (0, 0): (0, 1),
        (1, 0): (1, 0),
        (0, 1): (1, 0),
        (1, 1): (0, 1)
    }.items()

    for b, spec in cases:
        # Create oracle
        oracle = create_oracle(q0, q1, b)

        # Initialize circuit
        circuit = Circuit([q0, q1],
                          [
                              [H(q0), H(q1)],
                              oracle,
                              [H(q0), H(q1)],
                              measure(q0)
                          ])

        circuit.initialize([(1, 0), (0, 1)])

        # Symbolic execution
        (q0_final, _) = circuit.get_final_qbits()

        circuit.set_specification((q0_final, spec, SpecificationType.equality_pair))

        circuit.prove()


def create_oracle(q0: QbitVal, q1: QbitVal, output: Tuple[int, int]):
    """
    Create all four possible oracles.
    :param q0: first qubit.
    :param q1: second qubit.
    :param output: desired oracle output.
    :return: oracle.
    """
    if output == (0, 0):
        return [I(q0), I(q1)]
    if output == (0, 1):
        return [CNOT(q1, q0)]
    if output == (1, 0):
        return [X(q0), CNOT(q1, q0)]
    if output == (1, 1):
        return [X(q0)]


if __name__ == "__main__":
    prove_deutsch_algorithm()
