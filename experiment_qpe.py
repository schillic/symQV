import time

import numpy as np
from z3 import Real, Not, Implies

from symqv.lib.expressions.qbit import Qbits
from symqv.lib.models.circuit import Circuit, Method
from symqv.lib.operations.gates import H, R, SWAP, P
from symqv.lib.utils.arithmetic import complex_kron_n_ary


def create_qpe_circuit(n: int, theta):
    """
    Create QPE circuit.
    :param n: number of qbits.
    :return: circuit.
    """
    # Value of θ which appears in the definition of the unitary U above.
    # Try different values.
    qbits = Qbits([f'q{i}' for i in range(n)] + ['u'])

    u_bit = n

    program = []

    for j in range(n):
        program.append(H(qbits[j]))

    for j in range(n):
        program.append(P(qbits[u_bit], (2 * theta * np.pi) ** (2 ** (n - j - 1)))
                       .controlled_by(qbits[j]))

    # Inverse QFT
    # SWAP
    for j in range(int(np.floor(n / 2))):
        program.append(
            SWAP(qbits[j], qbits[n - (j + 1)]))

    # Rotations
    for j in range(n):
        for k in range(2, n - j + 1):
            program.append(R(qbits[j], k).controlled_by(qbits[j + k - 1]))

        program.append(H(qbits[j]))
    return Circuit(qbits, program)


def prove_qpe(n: int):
    theta = Real('theta')

    # Build circuit
    circuit = create_qpe_circuit(n, theta)

    circuit.initialize([(1, 0) for _ in range(n + 1)])

    # Build specification
    final_qbits = circuit.get_final_qbits()
    final_state = complex_kron_n_ary([qbit.to_complex_list() for qbit in final_qbits])

    circuit.solver.add(0 <= theta)
    circuit.solver.add(theta <= np.pi)

    a = Real('a')
    circuit.solver.add(a == 2 ** n * theta)

    probability = 4 / np.pi ** 2

    for j in range(2 ** n):
        circuit.solver.add(Not(Implies(a == j, final_state[j].r >= probability)))

    # Prove
    circuit.prove(method=Method.state_model,
                  overapproximation=False)


if __name__ == "__main__":
    for i in [3]:
        times = []

        for _ in range(5):
            start = time.time()
            prove_qpe(i)
            times.append(time.time() - start)

        print(f'Runtime for {i}:', np.mean(times))

