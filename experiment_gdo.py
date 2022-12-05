import time
from math import sqrt

import numpy as np
from z3 import And, Implies, Not

from symqv.lib.expressions.qbit import Qbits
from symqv.lib.models.circuit import Circuit, Method
from symqv.lib.operations.gates import H, Z, X
from symqv.lib.utils.arithmetic import complex_kron_n_ary


def prove_grover_diffuser_state_model(n: int, delta=0.0001):
    # Initialize circuit
    qbits = Qbits([f'q{i}' for i in range(n)])

    circuit = Circuit(qbits,
                      [
                          [H(qbit) for qbit in qbits],
                          [X(qbit) for qbit in qbits],
                          Z(qbits[0]).controlled_by(qbits[1:]),
                          [X(qbit) for qbit in qbits],
                          [H(qbit) for qbit in qbits],
                      ],
                      delta=delta)

    # Symbolic execution
    final_qbits = circuit.get_final_qbits()

    initial_state = complex_kron_n_ary([qbit.to_complex_list() for qbit in qbits])
    final_state = complex_kron_n_ary([qbit.to_complex_list() for qbit in final_qbits])

    # this specification checks if state vector entry i has negative phase,
    # then after diffusion, it should have a smaller value.
    conjunction = []

    for i in range(n):
        conjunction.append(Implies(
            initial_state[i].r <= -1 / sqrt(n),
            final_state[i].r <= initial_state[i].r))

    circuit.solver.add(Not(And(conjunction)))

    circuit.prove(method=Method.state_model,
                  dump_smt_encoding=False,
                  dump_solver_output=False,
                  overapproximation=False)


def prove_grover_diffuser(n: int, delta=0.0001):
    # Initialize circuit
    qbits = Qbits([f'q{i}' for i in range(n)])

    circuit = Circuit(qbits,
                      [
                          [H(qbit) for qbit in qbits],
                          [X(qbit) for qbit in qbits],
                          Z(qbits[0]).controlled_by(qbits[1:]),
                          [X(qbit) for qbit in qbits],
                          [H(qbit) for qbit in qbits],
                      ],
                      delta=delta)

    # Symbolic execution
    final_qbits = circuit.get_final_qbits()

    initial_state = complex_kron_n_ary(
        [qbit.to_complex_list() for qbit in qbits])
    final_state = complex_kron_n_ary(
        [qbit.to_complex_list() for qbit in final_qbits])

    # this specification checks if state vector entry i has negative phase,
    # then after diffusion, it should have a smaller value.
    conjunction = []

    for i in range(n):
        conjunction.append(Implies(
            initial_state[i].r <= -1 / sqrt(n),
            final_state[i].r <= initial_state[i].r))

    circuit.solver.add(Not(And(conjunction)))

    circuit.prove(method=Method.qbit_sequence_model,
                  overapproximation=True)


if __name__ == "__main__":
    for i in [5, 10]:
        times = []

        for _ in range(5):
            start = time.time()
            prove_grover_diffuser(i)
            times.append(time.time() - start)

        print(f'Runtime for {i}:', np.mean(times))
