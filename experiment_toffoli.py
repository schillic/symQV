import time

import numpy as np
from z3 import Implies, And, If, Not

from symqv.lib.expressions.qbit import Qbits
from symqv.lib.models.circuit import Circuit, Method
from symqv.lib.operations.gates import V, CNOT, V_dag


def prove_toffoli():
    # Initialize circuit
    a, b, c = Qbits(['a', 'b', 'c'])
    n = 3

    circuit = Circuit([a, b, c],
                      [
                          V(a).controlled_by(c),
                          V(a).controlled_by(b),
                          CNOT(b, c),
                          V_dag(a).controlled_by(b),
                          CNOT(b, c)
                      ])

    initial_values = [{(1, 0), (0, 1)} for _ in range(n)]

    circuit.initialize(initial_values)

    # Build specification
    final_qbits = circuit.get_final_qbits()

    circuit.solver.add(Implies(b.alpha.r == 1 and c.alpha.r == 1,
                               final_qbits[0].beta.r == a.alpha.r))

    # Prove
    circuit.prove(method=Method.qbit_sequence_model,
                  overapproximation=True)


# reordered gates, asserted solver result
def prove_toffoli_new():
    # Initialize circuit
    a, b, c = Qbits(['a', 'b', 'c'])
    n = 3

    circuit = Circuit([a, b, c],
                      [
                          V(c).controlled_by(a),
                          V(c).controlled_by(b),
                          CNOT(a, b),
                          V_dag(c).controlled_by(b),
                          CNOT(a, b)
                      ])

    initial_values = [{(1, 0), (0, 1)} for _ in range(n)]

    circuit.initialize(initial_values)

    # Build specification
    final_qbits = circuit.get_final_qbits()

    # both control qubits are 1 iff the third qubit gets negated
    circuit.solver.add(
        Not(  # negate specification
            If(
                And(a.beta.r == 1, b.beta.r == 1),
                And(c.alpha.r == final_qbits[2].beta.r, c.beta.r == final_qbits[2].alpha.r),
                And(c.alpha.r == final_qbits[2].alpha.r, c.beta.r == final_qbits[2].beta.r)
            )
        )
    )

    # Prove
    res = circuit.prove(method=Method.qbit_sequence_model,
                        # file_generation_only=True, dump_smt_encoding=True,
                        overapproximation=True)
    # print(res[0])
    assert res[0] == 'unsat'


if __name__ == "__main__":
    times = []

    for _ in range(5):
        start = time.time()
        prove_toffoli_new()
        times.append(time.time() - start)

    print(f'Runtime:', np.mean(times))
