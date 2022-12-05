from typing import List

from quavl.lib.expressions.qbit import Qbits, QbitVal
from quavl.lib.models.circuit import Circuit, Method
from quavl.lib.solver import SpecificationType
from quavl.lib.operations.gates import H, X, CNOT, I, Rx, Rz


def prove_deutsch_jozsa(n: int):
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
                              *oracle,
                              [H(q) for q in qbits],
                              [X(q) for q in qbits[:-1]],
                              X(qbits[-1]).controlled_by(qbits[:-1]),
                          ])

        print('Balanced:' if balanced else 'Constant:')
        print(circuit)

        circuit.initialize([(1, 0) for _ in qbits[:-1]] + [(0, 1)])

        # Symbolic execution
        final_qbits = circuit.get_final_qbits()

        print('standard oracle:', oracle)
        print('faulty oracle:', oracle[1:])

        if balanced:
            # balanced oracle: control qbits are not all (0, 1), so last qbit output is unchanged from (0, 1)
            circuit.set_specification((final_qbits[-1], (0, 1)), SpecificationType.equality_pair)
        else:
            # constant oracle: control qbits are all (0, 1), so last qbit output is changed to (1, 0)
            circuit.set_specification((final_qbits[-1], (1, 0)), SpecificationType.equality_pair)

        circuit.prove(method=Method.qbit_sequence_model,
                      overapproximation=True)


def repair_deutsch_jozsa(n: int):
    """
    Repair of a faulty version of the Deutsch-Jozsa algorithm.
    :param n: number of qbits.
    """
    qbits = Qbits([f'q{i}' for i in range(n)])

    balanced = True

    # Create oracles
    oracles = create_oracles(qbits, balanced)
    oracle = oracles[0]

    # Initialize circuit
    circuit = Circuit(qbits,
                      [
                          [H(q) for q in qbits],
                          *oracle[1:],
                          [H(q) for q in qbits],
                          [X(q) for q in qbits[:-1]],
                          X(qbits[-1]).controlled_by(qbits[:-1]),
                      ])

    print('Balanced:' if balanced else 'Constant:')
    print(circuit)

    circuit.initialize([(1, 0) for _ in qbits[:-1]] + [(0, 1)])

    # Symbolic execution
    final_qbits = circuit.get_final_qbits()

    print('standard oracle:', oracle)
    print('faulty oracle:', oracle[1:])

    # balanced oracle: control qbits are not all (0, 1), so last qbit output is unchanged from (0, 1)
    circuit.set_specification((final_qbits[-1], (0, 1)), SpecificationType.equality_pair)

    circuit.prove(method=Method.qbit_sequence_model,
                  synthesize_repair=True,
                  entangling_repair=True,
                  entangling_gate_index=1)


def verify_repaired_deutsch_jozsa(n: int):
    """
    Verify the repaired variant of the faulty Deutsch-Jozsa algorithm.
    :param n: number of qbits.
    """
    qbits = Qbits([f'q{i}' for i in range(n)])

    # repair outcome for balanced, oracle first gate removed, rotation [-pi, pi]
    rep_phi_0 = 3.1295
    rep_phi_1 = 3.1295
    rep_phi_2 = 3.1295
    rep_theta_0 = -3.1295
    rep_theta_1 = -3.1295
    rep_theta_2 = -3.1295

    balanced = True
    # Create oracles
    oracles = create_oracles(qbits, balanced)
    oracle = oracles[0]

    # Initialize circuit
    circuit = Circuit(qbits,
                      [
                          [H(q) for q in qbits],
                          *oracle[1:],
                          [H(q) for q in qbits],
                          [X(q) for q in qbits[:-1]],
                          X(qbits[-1]).controlled_by(qbits[:-1]),
                          Rx(qbits[0], rep_theta_0),
                          Rz(qbits[0], rep_phi_0),
                          Rx(qbits[1], rep_theta_1),
                          Rz(qbits[1], rep_phi_1),
                          Rx(qbits[2], rep_theta_2),
                          Rz(qbits[2], rep_phi_2)
                      ])

    print('Balanced:' if balanced else 'Constant:')
    print(circuit)

    circuit.initialize([(1, 0) for _ in qbits[:-1]] + [(0, 1)])

    # Symbolic execution
    final_qbits = circuit.get_final_qbits()

    if balanced:
        # balanced oracle: control qbits are not all (0, 1), so last qbit output is unchanged from (0, 1)
        circuit.set_specification((final_qbits[-1], (0, 1)), SpecificationType.equality_pair)
    else:
        # constant oracle: control qbits are all (0, 1), so last qbit output is changed to (1, 0)
        circuit.set_specification((final_qbits[-1], (1, 0)), SpecificationType.equality_pair)

    circuit.prove(method=Method.qbit_sequence_model,
                  dump_smt_encoding=False,
                  overapproximation=True)


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
    repair_deutsch_jozsa(n=3)
    verify_repaired_deutsch_jozsa(n=3)
