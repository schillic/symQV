import time
from typing import List

import numpy as np
from joblib import Parallel, delayed

from symqv.lib.expressions.qbit import QbitVal
from symqv.lib.expressions.qbit import Qbits
from symqv.lib.globals import precision_format
from symqv.lib.models.circuit import Circuit
from symqv.lib.models.measurement import Measurement
from symqv.lib.operations.gates import H, X, CNOT, I
from symqv.lib.operations import measure
from symqv.lib.utils.arithmetic import kron
from symqv.lib.solver import run_decision_procedure, SpecificationType

num_jobs = 4


def prove_deutsch_jozsa(n: int = 3, delta: float = 0.0001, parallel: bool = True):
    """
    Correctness proof of the Deutsch-Jozsa algorithm.
    :param n: number of qbits.
    :param parallel: true or false.
    :param delta: error bound.
    """
    qbits = Qbits([f'q{i}' for i in range(n)])

    for b in [True, False]:
        # Create oracles
        oracles = create_oracles(qbits, b)
        oracle = oracles[0]

        program = [
            [H(q) for q in qbits],
            oracle,
            [H(q) for q in qbits],
            measure(qbits[:-1])
        ]

        # 1 Calculate number of measurement branches
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
            final_qbits = circuit.get_final_qbits()

            spec = kron([np.array([[1], [0]]) for _ in qbits[:-1]] + [np.array([[0], [1]])])
            spec = spec.T[0]

            if b:
                circuit.set_specification(spec, SpecificationType.final_state_vector)
            else:
                circuit.set_specification(spec, SpecificationType.final_state_vector, is_equality_specification=False)

            (temp_file, qbit_identifiers) = circuit.prove(measurement_branch=i, file_generation_only=True)
            job_objects.append((temp_file.name, qbit_identifiers, circuit.specification))

        print('Starting parallel proof...\n')

        if parallel:
            results = Parallel(n_jobs=num_jobs)(
                delayed(run_decision_procedure)(temp_file_name, qbit_identifiers, None, specification, None, 0.0001) for
                (temp_file_name, qbit_identifiers, specification) in job_objects)
        else:
            for (temp_file_name, qbit_identifiers, specification) in job_objects:
                results.append(
                    run_decision_procedure(temp_file_name, qbit_identifiers, None, specification, None, 0.0001))

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
    if len(qbits) == 2:
        return create_oracles_2_qubits(qbits[0], qbits[1], balanced)

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


def create_oracles_2_qubits(q0: QbitVal, q1: QbitVal, balanced: bool):
    """
    Create all four possible oracles.
    :param q0: first qubit.
    :param q1: second qubit.
    :param output: desired oracle output.
    :return: oracle.
    """
    if balanced:
        return [[I(q0), I(q1)], [X(q0)]]
    else:
        return [[CNOT(q1, q0)], [X(q0), CNOT(q1, q0)]]


if __name__ == "__main__":
    prove_deutsch_jozsa(6, 0.0001)
