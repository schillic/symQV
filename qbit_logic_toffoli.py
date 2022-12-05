from z3 import Implies

from z3 import Implies

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
                  dump_smt_encoding=True,
                  dump_solver_output=True,
                  overapproximation=True)


if __name__ == "__main__":
    prove_toffoli()

# Positive
# Naïve     No overapproximation  symQV
# 5.9602    2.2387                0.7278

# Negative
# Naïve     No overapproximation  symQV
# 4.6495    1.2916                0.4558
