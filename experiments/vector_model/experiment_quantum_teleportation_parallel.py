import time
from typing import List

import numpy as np
from joblib import Parallel, delayed

from quavl.lib.constants import CNOT_matrix, H_matrix, I_matrix, SWAP_matrix, CZ_matrix, X_matrix, Z_matrix, \
    CNOT_reversed_matrix
from quavl.lib.expressions.qbit import QbitVal, Qbit
from quavl.lib.expressions.qbit import Qbits
from quavl.lib.globals import precision_format
from quavl.lib.models.circuit import Circuit
from quavl.lib.models.measurement import Measurement
from quavl.lib.operations.gates import H, X, CNOT, I, SWAP, CZ, Z
from quavl.lib.operations.measurements import measure
from quavl.lib.utils.arithmetic import kron, matmul
from quavl.lib.solver import run_decision_procedure, SpecificationType

num_jobs = 1


def prove_quantum_teleportation(num_qbits: int = 0, delta: float = 0.0001, parallel: bool = True):
    """
    Correctness proof of the Quantum Teleportation Protocol in parallel.
    :param num_qbits: number of qbits.
    :param parallel: true or false.
    :param delta: error bound.
    """
    # Initialize circuit
    if num_qbits >= 3:
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

        program = [
            CNOT(psi, b0),
            H(psi),
            CNOT(b0, b1),
            SWAP(psi, b0),
            CZ(b0, b1),
            SWAP(psi, b0),
            measure([psi, b0])
        ]

        circuit = Circuit(qbits, program)

        circuit.initialize([None, (1, 0), (1, 0)])

        circuit.set_initial_gate_applications([
            H(b0),
            CNOT(b0, b1)
        ])

        # Symbolic execution
        for _ in range(num_qbits):
            spec = np.kron(spec, I_matrix)

        circuit.set_specification(spec, SpecificationType.transformation_matrix)
    elif num_qbits == 2:
        # Initialize circuit
        qbits = Qbits(['psi', 'b0'])

        psi = qbits[0]
        b0 = qbits[1]

        program = [
            CNOT(psi, b0),
            H(psi),
            CNOT(b0, psi),
            SWAP(psi, b0),
            CZ(psi, b0),
            SWAP(psi, b0),
            measure(b0)
        ]

        circuit = Circuit([psi, b0], program)

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
    elif num_qbits == 1:
        # Initialize circuit
        qbits = Qbits(['psi'])
        psi = qbits[0]

        program = [
            X(psi),
            H(psi),
            X(psi),
            X(psi),
            Z(psi),
            X(psi),
            measure(psi)
        ]

        circuit = Circuit(qbits, program)

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
    else:
        raise Exception("Unsupported qbit count.")

    # Calculate number of measurement branches
    measurements = [operation for operation in program if isinstance(operation, Measurement)]

    num_measurements = 0

    for measurement in measurements:
        if isinstance(measurement.arguments, QbitVal):
            num_measurements += 1
        else:
            num_measurements += len(measurement.arguments)

    num_branches = 2 ** num_measurements

    print('Preprocessing...\n')
    start_parallel = time.time()

    job_objects = []
    results = []

    # 2 solve each
    for i in range(num_branches):
        # Initialize circuit
        circuit = Circuit(qbits, program, delta)

        circuit.initialize([(1, 0) for _ in qbits[:-1]] + [(0, 1)])

        # Symbolic execution
        (temp_file, qbit_identifiers) = circuit.prove(measurement_branch=i, file_generation_only=True)
        job_objects.append((temp_file.name, qbit_identifiers, circuit.specification))

    print('Starting parallel proof...\n')

    if parallel:
        results = Parallel(n_jobs=num_jobs)(
            delayed(run_decision_procedure)(temp_file_name, qbit_identifiers, None, specification, None, 0.0001) for
            (temp_file_name, qbit_identifiers, specification) in job_objects)
    else:
        for (temp_file_name, qbit_identifiers, specification) in job_objects:
            results.append(run_decision_procedure(temp_file_name, qbit_identifiers, None, specification, None, 0.0001))

    end_parallel = time.time()
    time_parallel = end_parallel - start_parallel

    print(f'\nParallelized elapsed time {precision_format.format(time_parallel)} seconds.')


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
    prove_quantum_teleportation(1, 0.0001)
