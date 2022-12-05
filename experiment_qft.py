from typing import List

import numpy as np
from z3 import If, Or, Not, And

from quavl.lib.constants import cos, sin, pi
from quavl.lib.expressions.complex import ComplexVal
from quavl.lib.expressions.qbit import Qbits, QbitVal
from quavl.lib.models.circuit import Circuit, Method
from quavl.lib.operations.gates import H, R, SWAP, Rx, Rz
from quavl.lib.solver import SpecificationType


def build_qft_specification(qbits: List[QbitVal]):
    """
    Build specification for QFT.
    :param qbits: qbits.
    :return: specification.
    """
    n = len(qbits)

    output_bit_values = [If(q.beta.r == 1.0, 1.0, 0.0) for q in qbits]

    output_bit_fractions = []

    for i in range(n):
        output_bit_fraction = 0.0

        for j, k in enumerate(range(i, n)):
            output_bit_fraction += output_bit_values[k] * (1 / (2 ** (j + 1)))

        output_bit_fractions.append(output_bit_fraction)

    inv_root2 = 1 / np.sqrt(2)  # inv_root2 = 1 / Sqrt(Real(2))

    output_specification = [QbitVal(
        alpha=ComplexVal(r=inv_root2),
        beta=ComplexVal(r=inv_root2 * cos(2 * pi * v),
                        i=inv_root2 * sin(2 * pi * v)))
        for v in output_bit_fractions]

    return output_specification


def prove_qft():
    """
    Correctness proof of the QFT.
    """
    # Initialize circuit
    q0, q1, q2 = Qbits(['q0', 'q1', 'q2'])
    n = 3

    circuit = Circuit([q0, q1, q2],
                      [
                          H(q0),
                          R(q0, 2).controlled_by(q1),
                          R(q0, 3).controlled_by(q2),
                          H(q1),
                          R(q1, 2).controlled_by(q2),
                          H(q2),
                          SWAP(q0, q2)
                      ])

    initial_values = [{(1, 0), (0, 1)} for _ in range(n)]

    circuit.initialize(initial_values)

    # Build specification
    final_qbits = circuit.get_final_qbits()
    spec_qbits = Qbits([f'spec_q{i}' for i in range(n)])

    output_specification = build_qft_specification(circuit.qbits)

    conjunction = []

    for i in range(n):
        conjunction.append(spec_qbits[i] == output_specification[i])

    circuit.solver.add(And(conjunction))

    circuit.final_qbits += spec_qbits

    # Set specification
    circuit.set_specification([(final_qbits[i], spec_qbits[i]) for i in range(n)],
                              SpecificationType.equality_pair_list)

    # Prove and repair
    circuit.prove(method=Method.qbit_sequence_model,
                  dump_solver_output=True,
                  synthesize_repair=True)


def repair_phase_error_qft():
    """
    Repair of a faulty QFT where a gate is omitted.
    """
    # Initialize circuit
    q0, q1, q2 = Qbits(['q0', 'q1', 'q2'])
    n = 3

    circuit = Circuit([q0, q1, q2],
                      [
                          H(q0),
                          R(q0, 2).controlled_by(q1),
                          R(q0, 3).controlled_by(q2),
                          H(q1),
                          R(q1, 2).controlled_by(q2),
                          H(q2),
                          SWAP(q0, q2)
                      ])

    initial_values = [{(1, 0), (0, 1)} for _ in range(n)]

    circuit.initialize(initial_values)

    # Build specification
    final_qbits = circuit.get_final_qbits()
    spec_qbits = Qbits([f'spec_q{i}' for i in range(n)])

    output_specification = build_qft_specification(circuit.qbits)

    conjunction = []

    for i in range(n):
        conjunction.append(spec_qbits[i] == output_specification[i])

    circuit.solver.add(And(conjunction))

    circuit.final_qbits += spec_qbits

    # Set specification
    circuit.set_specification([(final_qbits[i], spec_qbits[i]) for i in range(n)],
                              SpecificationType.equality_pair_list)

    # Prove and repair
    circuit.prove(method=Method.qbit_sequence_model,
                  dump_solver_output=True,
                  synthesize_repair=True,
                  entangling_repair=True)


def repair_omission_qft():
    """
    Repair of a faulty QFT where a gate is omitted.
    """
    # Initialize circuit
    q0, q1, q2 = Qbits(['q0', 'q1', 'q2'])
    n = 3

    circuit = Circuit([q0, q1, q2],
                      [
                          R(q0, 2).controlled_by(q1),
                          R(q0, 3).controlled_by(q2),
                          H(q1),
                          R(q1, 2).controlled_by(q2),
                          H(q2),
                          SWAP(q0, q2)
                      ])

    initial_values = [{(1, 0), (0, 1)} for _ in range(n)]

    circuit.initialize(initial_values)

    # Build specification
    final_qbits = circuit.get_final_qbits()
    spec_qbits = Qbits([f'spec_q{i}' for i in range(n)])

    output_specification = build_qft_specification(circuit.qbits)

    conjunction = []

    for i in range(n):
        conjunction.append(spec_qbits[i] == output_specification[i])

    circuit.solver.add(And(conjunction))

    circuit.final_qbits += spec_qbits

    # Set specification
    circuit.set_specification([(final_qbits[i], spec_qbits[i]) for i in range(n)],
                              SpecificationType.equality_pair_list)

    # Prove and repair
    circuit.prove(method=Method.qbit_sequence_model,
                  dump_solver_output=True,
                  synthesize_repair=True,
                  entangling_repair=True,
                  entangling_gate_index=0)


def prove_repaired_qft_phase_error():
    """
    Prove the correctness of the repaired QFT with phase error.
    :return:
    """
    # Initialize circuit
    q0, q1, q2 = Qbits(['q0', 'q1', 'q2'])
    n = 3

    # repair outcome: rotation, [-0.1, 0.1]
    rep_theta_0 = [0.0140625, 0.014062500000000002]
    rep_phi_0 = [-0.014062500000000002, -0.0140625]

    rep_theta_1 = [0.0140625, 0.014062500000000002]
    rep_phi_1 = [-0.014062500000000002, -0.0140625]

    rep_theta_2 = [0.0140625, 0.014062500000000002]
    rep_phi_2 = [-0.014062500000000002, -0.0140625]

    circuit = Circuit([q0, q1, q2],
                      [
                          H(q0),
                          R(q0, 2).controlled_by(q1),
                          R(q0, 3).controlled_by(q2),
                          H(q1),
                          R(q1, 2).controlled_by(q2),
                          H(q2),
                          SWAP(q0, q2),
                          Rx(q0, np.mean(rep_theta_0)),
                          Rz(q0, np.mean(rep_phi_0)),
                          Rx(q1, np.mean(rep_theta_1)),
                          Rz(q1, np.mean(rep_phi_1)),
                          Rx(q2, np.mean(rep_theta_2)),
                          Rz(q2, np.mean(rep_phi_2))
                      ])

    initial_values = [{(1, 0), (0, 1)} for _ in range(n)]

    circuit.initialize(initial_values)

    # Build specification
    final_qbits = circuit.get_final_qbits()
    spec_qbits = Qbits([f'spec_q{i}' for i in range(n)])

    output_specification = build_qft_specification(circuit.qbits)

    disjunction = []

    for i in range(n):
        disjunction.append(Not(spec_qbits[i] == output_specification[i]))

    circuit.solver.add(Or(disjunction))

    circuit.final_qbits += spec_qbits

    # Set specification
    circuit.set_specification([(final_qbits[i], spec_qbits[i]) for i in range(n)],
                              SpecificationType.equality_pair_list)

    # Prove and repair
    circuit.prove(method=Method.qbit_sequence_model,
                  overapproximation=True)


def prove_repaired_qft_rotation_large_interval():
    """
    Prove the correctness of the repaired QFT with the large interval rotation.
    :return:
    """
    # Initialize circuit
    q0, q1, q2 = Qbits(['q0', 'q1', 'q2'])
    n = 3

    rep_theta_0 = [1.9148873922774943, 1.9148873922774945]
    rep_phi_0 = [-1.9148873922774945, -1.9148873922774943]

    rep_theta_1 = [1.9148873922774943, 1.9148873922774945]
    rep_phi_1= [-1.9148873922774945, -1.9148873922774943]

    rep_theta_2 = [1.9148873922774943, 1.9148873922774945]
    rep_phi_2 = [-1.9148873922774945, -1.9148873922774943]

    circuit = Circuit([q0, q1, q2],
                      [
                          R(q0, 2).controlled_by(q1),
                          R(q0, 3).controlled_by(q2),
                          H(q1),
                          R(q1, 2).controlled_by(q2),
                          H(q2),
                          SWAP(q0, q2),
                          Rx(q0, np.mean(rep_theta_0)),
                          Rz(q0, np.mean(rep_phi_0)),
                          Rx(q1, np.mean(rep_theta_1)),
                          Rz(q1, np.mean(rep_phi_1)),
                          Rx(q2, np.mean(rep_theta_2)),
                          Rz(q2, np.mean(rep_phi_2))
                      ])

    initial_values = [{(1, 0), (0, 1)} for _ in range(n)]

    circuit.initialize(initial_values)

    # Build specification
    final_qbits = circuit.get_final_qbits()
    spec_qbits = Qbits([f'spec_q{i}' for i in range(n)])

    output_specification = build_qft_specification(circuit.qbits)

    disjunction = []

    for i in range(n):
        disjunction.append(Not(spec_qbits[i] == output_specification[i]))

    circuit.solver.add(Or(disjunction))

    circuit.final_qbits += spec_qbits

    # Set specification
    circuit.set_specification([(final_qbits[i], spec_qbits[i]) for i in range(n)],
                              SpecificationType.equality_pair_list)

    # Prove and repair
    circuit.prove(method=Method.qbit_sequence_model,
                  overapproximation=True)


if __name__ == "__main__":
    prove_qft()
    repair_phase_error_qft()
    repair_phase_error_qft()
    prove_repaired_qft_phase_error()
    prove_repaired_qft_rotation_large_interval()
