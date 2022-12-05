import getopt
import sys

from symqv.lib.constants import CNOT_matrix, I_matrix, H_matrix, SWAP_matrix, CZ_matrix, CNOT_reversed_matrix, X_matrix, Z_matrix
from symqv.lib.expressions.qbit import Qbits, Qbit
from symqv.lib.models.circuit import Circuit
from symqv.lib.solver import SpecificationType
from symqv.lib.operations.gates import CNOT, H, CZ, SWAP, X, custom_gate, Z
from symqv.lib.operations.measurements import measure
from symqv.lib.utils.arithmetic import kron, matmul
import numpy as np


def scalability_experiment(num_qbits: int, num_gates: int, measurement: bool = True, optimized: bool = False):
    """
    Scalability (based on Quantum teleportation)
    :param num_qbits: number of qbits to add.
    :param num_gates: number gates by which to extend the quantum teleportation protocol circuit by (factor 2).
    :param optimized: collapsing operations.
    :return: None.
    """
    if num_qbits == -1:
        scalability_width_2()
        return

    if num_qbits == -2:
        scalability_width_1()
        return

    # Initialize circuit
    (psi, b0, b1) = Qbits(['psi', 'b0', 'b1'])

    qbits = list((psi, b0, b1))

    for i in range(num_qbits):
        qbits.append(Qbit(f'q_{i}'))

    spec = matmul(
        [
            kron([CNOT_matrix, I_matrix]),
            kron([H_matrix, I_matrix, I_matrix]),
            kron([I_matrix, CNOT_matrix]),
            kron([SWAP_matrix, I_matrix]),
            kron([I_matrix, CZ_matrix]),
            kron([SWAP_matrix, I_matrix])
        ])

    if optimized:
        program = [
            custom_gate('Quantum Teleport', [psi, b0, b1], spec)
        ]
    else:
        program = [
            CNOT(psi, b0),
            H(psi),
            CNOT(b0, b1),
            SWAP(psi, b0),
            CZ(b0, b1),
            SWAP(psi, b0)
        ]

    for i in range(num_gates):
        program.extend([X(psi),
                        X(psi)])

    if measurement:
        program.extend([
            measure(psi),
            measure(b0)])

    circuit = Circuit(qbits, program)

    circuit.initialize([None, (1, 0), (1, 0)])

    circuit.set_initial_gate_applications([
        H(b0),
        CNOT(b0, b1)
    ])

    # Symbolic execution
    final_qbits = circuit.get_final_qbits()

    if measurement:
        circuit.set_specification((psi, final_qbits[2]), SpecificationType.equality_pair)
    else:
        for _ in range(num_qbits):
            spe, c = np.kron(spec, I_matrix)

        circuit.set_specification(spec, SpecificationType.transformation_matrix)

    circuit.prove()


def scalability_width_2():
    """
    Scalability (based on Quantum teleportation), downsized to 2 qubits.
    """
    # Initialize circuit
    (psi, b0) = Qbits(['psi', 'b0'])

    circuit = Circuit([psi, b0],
                      [
                          CNOT(psi, b0),
                          H(psi),
                          CNOT(b0, psi),
                          SWAP(psi, b0),
                          CZ(psi, b0),
                          SWAP(psi, b0)
                      ])

    circuit.initialize([None, (1, 0)])

    circuit.set_initial_gate_applications([
        H(b0),
        CNOT(b0, psi)
    ])

    # Symbolic execution
    circuit.set_specification(matmul(
        [
            kron([CNOT_matrix]),
            kron([H_matrix, I_matrix]),
            kron([CNOT_reversed_matrix]),
            kron([SWAP_matrix]),
            kron([CZ_matrix]),
            kron([SWAP_matrix])
        ]),
        SpecificationType.transformation_matrix)
    circuit.prove()


def scalability_width_1():
    """
    Scalability (based on Quantum teleportation), downsized to 1 qubit.
    """
    # Initialize circuit
    psi = Qbit('psi')

    circuit = Circuit([psi],
                      [
                          X(psi),
                          H(psi),
                          X(psi),
                          X(psi),
                          Z(psi),
                          X(psi)
                      ])

    # Symbolic execution
    circuit.set_specification(matmul(
        [
            kron([X_matrix]),
            kron([H_matrix]),
            kron([X_matrix]),
            kron([X_matrix]),
            kron([Z_matrix]),
            kron([X_matrix])
        ]),
        SpecificationType.transformation_matrix)
    circuit.prove()


def main(argv):
    """
    Read command line arguments.
    :param argv:
    :return:
    """
    qbits = ''
    gates = ''
    measurement = ''

    try:
        opts, args = getopt.getopt(argv, "q:g:m:", ["qbits=", "gates=", "measurement="])
    except getopt.GetoptError:
        print('test.py -q <qbits>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -g <gates> -q <qbits>')
            sys.exit()
        elif opt in ("-q", "--qbits"):
            qbits = arg
        elif opt in ("-g", "--gates"):
            gates = arg
        elif opt in ("-m", "--measurement"):
            measurement = arg

    print('================================')
    print(f'qbits: {qbits}, gates: {gates}, measurement: {measurement}')

    scalability_experiment(int(qbits), int(gates), bool(int(measurement)))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1:])
    else:
        scalability_experiment(num_qbits=3, num_gates=0, measurement=True, optimized=False)
