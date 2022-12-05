from typing import Tuple, List, Callable

import numpy as np
from scipy import optimize

from symqv.lib.constants import zero, one
from symqv.lib.models.hybrid_circuit import HybridCircuit
from symqv.lib.models.specification import Specification
from symqv.lib.operations.state_decomposition import from_angles, separate_kth_qbit_from_state, euclidean_distance
from symqv.lib.utils.arithmetic import kron


def verify(qc: HybridCircuit, specification: Specification) -> Tuple[bool, np.array]:
    """
    Verify a hybrid quantum program against an input-output specification.
    :param qc: hybrid quantum circuit.
    :param specification: input-output specification.
    :return: Verification result, and counterexample if violated.
    """
    if all([input.is_symbolic() for input in specification.inputs]):
        # Specification with a symbolic input

        if specification.inputs[0].right == 'any':
            # Any is the complete input state space
            bounds = [(0, 2 * np.pi), (0, np.pi)]
        elif specification.inputs[0].right == 'bitvector':
            # bitvector uses combinatorial optimization
            for case in range(2 ** qc.qbits):
                # for each case
                initial_state_qbits = []
                bitvector = ('{:0' + str(qc.qbits) + 'b}').format(case)

                for bit in bitvector:
                    if bit == '0':
                        initial_state_qbits.append(zero)
                    else:
                        initial_state_qbits.append(one)

                initial_state = kron(initial_state_qbits)

                final_states = qc.get_final_states(initial_state)

                for final_state in final_states:
                    spec_final_state_qbits = [zero] * len(specification.outputs)

                    for i, output in enumerate(specification.outputs):
                        sub_bitvector = bitvector[output.right.start:output.right.end]
                        x = int(sub_bitvector, 2)

                        output_calculation = eval(output.right.function)

                        spec_final_state_qbits[i] = output_calculation

                    spec_final_state = kron(spec_final_state_qbits)

                    if not np.allclose(spec_final_state, final_state):
                        return False, bitvector

            return True, None
        else:
            # Reduced input state space currently unsupported
            raise Exception(f'Symbolic input bounds {specification.inputs.right} not supported')

        # Optimize the error function
        objective_function = build_objective_function(qc, specification)
        result = optimize.dual_annealing(objective_function, bounds)

        print(result)

        # If the minimum is zero, there is no violation
        if np.isclose(result.fun, 0):
            return True, None

        # If the minimum is below zero, we have found a counter-example.
        phi, theta = result.x[0], result.x[1]
        counterexample = from_angles(phi, theta)

        return False, counterexample
    else:
        # Concrete input specification currently unsupported.
        raise Exception('Unsupported specification format.')


def build_objective_function(qc: HybridCircuit, specification: Specification) -> Callable:
    """
    Build an objective function.
    :param qc: hybrid quantum circuit.
    :param specification: specification.
    :return: Objective function.
    """

    def objective_function(x: List[float]):
        """
        Objective function for verifying a quantum program with a symbolic input.
        It is zero exactly when the desired output is matched, and negative when not.
        :param x: phi and theta.
        :return: objective value (negative euclidean distance between desired and actual output).
        """
        phi = x[0]
        theta = x[1]
        psi = from_angles(phi, theta)

        initial_state_qbits = [zero] * qc.qbits
        initial_state_qbits[specification.inputs[0].left] = psi

        initial_state = kron(initial_state_qbits)

        final_states = qc.get_final_states(initial_state)

        distances = []

        print(final_states)

        for final_state in final_states:
            output_qbit, separable, result = separate_kth_qbit_from_state(final_state,
                                                                          specification.outputs[0].left)

            if not separable:
                raise Exception('Entangled qbit.\n' + str(result))

            distances.append(euclidean_distance(output_qbit, psi))

        return -np.sum(distances)

    return objective_function
