import time
from math import sqrt

import numpy as np
from z3 import And, Implies, Not

from quavl.lib.expressions.qbit import Qbits
from quavl.lib.models.circuit import Circuit, \
    Method
from quavl.lib.operations.gates import H, Z, X
from quavl.lib.utils.arithmetic import \
    complex_kron_n_ary

n = [12]


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

    circuit.prove(
        method=Method.state_model,
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

    circuit.prove(
        method=Method.qbit_sequence_model,
        dump_smt_encoding=False,
        dump_solver_output=False,
        overapproximation=True)


if __name__ == "__main__":
    prove_grover_diffuser_state_model(5)

    for i in n:
        times = []

        for _ in range(10):
            start = time.time()
            prove_grover_diffuser(i)
            times.append(time.time() - start)

        print(f'Runtime for {i}:', np.mean(times))

# n  time     vars
# 3  12.8167  42
# 4  21.5770  56
# 5  35.1327  70
# 6  52.6286  84
# 10 161.4223 140
# 20 831.4069 280
# 30 2634.3716 420

#     naïve          abstraction
# 4   37.6322  280
# 5                  56.7393    350


# Grover diffuser
# no approximation
# n  time       vars
# 5  9.1814     280
# 10 177.4      560
# 12 697.6627   672
# 15            840

# with approximation
# n  time       vars
# 5  2.4363     280
# 10 16.9517    560
# 12 29.2956    672
# 15 58.0624    840
# 18 99.2628    1008
# 20 132.5820   1120
# 24

# Phobos 124 runtimes (w/ approximation)
# n  time
# 5  1.3175
# 10
# 12 20.2044
# 15 61.6502
# 18 291.4152
# 20 1025.1401
# 22 3959.1768   1232
# 23 7863.4620
# 24* 14955.6860487399

# Phobos 124 runtimes (w/o approximation)
# 5  7.068533182144165
# 10 190.01711893081665
# 12 854.6622030735016
# 15 10287.955570459366


# Deltas:
# Runtime for 15, delta = 0.001: 130.9506857395172
# Runtime for 15, delta = 0.0001: 1794.4355990886688
# Runtime for 15, delta = 1e-05: 1986.9527883529663
# Runtime for 15, delta = 1e-06: 2444.958612203598
# Runtime for 15, delta = 1e-07: 2961.9769761562347
# Runtime for 15, delta = 1e-08: 3035.2942407131195

# Runtime for 12, delta = 0.001: 66.94435577392578
# Runtime for 12, delta = 0.0001: 65.97293851375579
# Runtime for 12, delta = 1e-05: 67.07310202121735
# Runtime for 12, delta = 1e-06: 66.20236582756043

# 20.5 20.2 20.5 20.3

# Runtime for 18, delta = 0.001: 1074.9715242385864
# Runtime for 18, delta = 0.0001: 1058.5109159946442
# Runtime for 18, delta = 1e-05: 1122.9925739765167
# Runtime for 18, delta = 1e-06: 1681.63667345047
# Runtime for 18, delta = 1e-08: 3525.284265756607
# Runtime for 18, delta = 1e-09: 3482.0019290447235
# Runtime for 18, delta = 1e-10: 4305.233973741531
