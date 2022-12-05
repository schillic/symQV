from z3 import Implies

import numpy as np
from z3 import Implies

from quavl.lib.expressions.qbit import Qbits
from quavl.lib.models.circuit import Circuit, Method
from quavl.lib.operations.gates import Rx, Rz, V, CNOT, V_dag


def repair_toffoli():
    # Initialize circuit
    a, b, c = Qbits(['a', 'b', 'c'])
    n = 3

    circuit = Circuit([a, b, c],
                      [
                          V(a).controlled_by(c),
                          V(a).controlled_by(b),
                          # CNOT(b, c),
                          V_dag(a).controlled_by(b),
                          CNOT(b, c)
                      ])

    initial_values = [{(1, 0), (0, 1)} for _ in range(n)]

    circuit.initialize(initial_values)

    # Build specification
    final_qbits = circuit.get_final_qbits()

    circuit.solver.add(Implies(b.alpha.r == 1 and c.alpha.r == 1,
                               final_qbits[0].beta.r == a.alpha.r))

    # Prove and repair
    circuit.prove(method=Method.qbit_sequence_model,
                  dump_solver_output=True,
                  synthesize_repair=True)


def prove_repaired_toffoli():
    rep_theta_0 = [-0.10000000000000001, -0.099999999999999992]
    rep_phi_0 = [0.099999999999999992, 0.10000000000000001]

    rep_theta_1 = [-0.10000000000000001, -0.099999999999999992]
    rep_phi_1 = [0.099999999999999992, 0.10000000000000001]

    rep_theta_2 = [-0.10000000000000001, -0.099999999999999992]
    rep_phi_2 = [0.099999999999999992, 0.10000000000000001]

    # Initialize circuit
    a, b, c = Qbits(['a', 'b', 'c'])
    n = 3

    circuit = Circuit([a, b, c],
                      [
                          V(a).controlled_by(c),
                          V(a).controlled_by(b),
                          # CNOT(b, c),
                          V_dag(a).controlled_by(b),
                          CNOT(b, c),
                          Rx(a, np.mean(rep_theta_0)),
                          Rz(a, np.mean(rep_phi_0)),
                          Rx(b, np.mean(rep_theta_1)),
                          Rz(b, np.mean(rep_phi_1)),
                          Rx(c, np.mean(rep_theta_2)),
                          Rz(c, np.mean(rep_phi_2))
                      ])

    initial_values = [{(1, 0), (0, 1)} for _ in range(n)]

    circuit.initialize(initial_values)

    # Build specification
    final_qbits = circuit.get_final_qbits()

    circuit.solver.add(Implies(b.alpha.r == 1 and c.alpha.r == 1,
                               final_qbits[0].beta.r == a.alpha.r))

    # Prove and repair
    circuit.prove(method=Method.qbit_sequence_model,
                  dump_solver_output=True,
                  synthesize_repair=True)


if __name__ == "__main__":
    repair_toffoli()
    prove_repaired_toffoli()
