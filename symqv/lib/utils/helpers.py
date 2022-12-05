from functools import reduce
from typing import List, Union, Callable

import numpy as np

from symqv.lib.constants import I_matrix, SWAP_matrix
from symqv.lib.expressions.complex import ComplexVal
from symqv.lib.expressions.qbit import QbitVal
from symqv.lib.globals import precision_format
from symqv.lib.utils.arithmetic import matmul, kron

pi = '(acos -1.0)'  # round(math.pi, 10) #3.141593


def get_qbit_indices(system_qbits: List[Union[QbitVal, str]], gate_qbits: List[Union[QbitVal, str]]) -> List[int]:
    """
    Get qbit indices within a system of qbits.
    :param gate_qbits: qbits from which the indices to get from.
    :param system_qbits: all qbits in the system (order is important here).
    :return: the gate qbits' indices within the system.
    """
    qbit_indices = []

    for q in gate_qbits:
        for i in range(len(system_qbits)):
            qbit_identifier = q.get_identifier() if isinstance(q, QbitVal) else q

            system_qbit = system_qbits[i]
            system_qbit_identifier = system_qbit.get_identifier() if isinstance(system_qbit, QbitVal) else system_qbit

            if qbit_identifier == system_qbit_identifier:
                qbit_indices.append(i)

    return qbit_indices


def are_qbits_reversed(qbit_indices: List[int]) -> bool:
    """
    Checks if qbit indices are reversed.
    :param qbit_indices: list of qbit indices.
    :return: true if reversed, false otherwise.
    """
    return qbit_indices[-1] < qbit_indices[0]


def are_qbits_adjacent(qbit_indices: List[int]) -> bool:
    """
    Checks if qbit indices are adjacent.
    :param qbit_indices: list of qbit indices.
    :return: true if adjacent, false otherwise.
    """
    if len(qbit_indices) == 1:
        # Trivially true
        return True

    if len(qbit_indices) > 2:
        for i in range(len(qbit_indices[:-1])):
            if qbit_indices[i] != qbit_indices[i + 1] - 1:
                return False
        return True

    return qbit_indices[0] == qbit_indices[1] - 1 or qbit_indices[1] == qbit_indices[0] - 1


def identity_pad_gate(gate: np.ndarray, gate_qbits: List[int], num_qbits: int) -> np.ndarray:
    """
    Pad a quantum gate with identity matrices using the Kronecker product.
    :param gate: gate to be padded.
    :param gate_qbits: qbit indices of the gate.
    :param num_qbits: number of qbits in the system.
    :return: Identity padded quantum gate.
    """
    if gate is None:
        raise Exception('Gate cannot be None.')

    if any(i >= num_qbits for i in gate_qbits):
        raise Exception('Qbit index higher than qbit count.')

    for i in range(len(gate_qbits)):
        if 0 < i < len(gate_qbits) - 1 \
                and gate_qbits[i] != gate_qbits[i - 1] + 1 \
                and gate_qbits[i] != gate_qbits[i + 1] + 1:
            raise Exception("Gates can only be applied to neighboring qbits.")

    kronecker_factors = []
    gate_added = False

    for i in range(num_qbits):
        if i in gate_qbits:
            if not gate_added:
                kronecker_factors.append(gate)
                gate_added = True
        else:
            kronecker_factors.append(I_matrix)

    identity_padded_gate = kron(kronecker_factors)

    return identity_padded_gate


def identity_pad_single_qbit_gates(gates: List[np.ndarray], gate_qbits: List[int], num_qbits: int) -> np.ndarray:
    """
    Combine multiple single qbit gates while identity padding them.
    :param gates: single qbit gate list.
    :param gate_qbits: list of qbit indices each gate acts on.
    :param num_qbits: number of qbits in the system.
    :return: Identity padded product of single qbit gates.
    """
    if len(gates) != len(gate_qbits):
        raise Exception("Each gate needs exactly one qbit to act on.")

    single_gates = []

    for i in range(len(gates)):
        single_gates.append(identity_pad_gate(gates[i], [gate_qbits[i]], num_qbits))

    return reduce((lambda x, y: np.matmul(x, y)), single_gates)


def swap_transform_non_adjacent_gate(gate: np.ndarray, gate_qbits: List[int], num_qbits: int) -> np.ndarray:
    """
    Use SWAP gates to apply a two-qbit gate to non-adjacent qbits.
    :param gate: two-qbit gate meant for two non-adjacent qbits.
    :param gate_qbits: the indices of the qbits to which the gate should be applied to.
    :param num_qbits: number of qbits in circuit.
    :return: Matrix that realizes the gate application to the non-adjacent qbits.
    """
    if gate_qbits[0] > gate_qbits[1]:
        raise Exception("Reverse is not supported.")

    num_gate_qbits = gate_qbits[1] - gate_qbits[0] + 1
    swap_gates = []

    for i in range(num_gate_qbits - 1):
        swap_gates.append(identity_pad_gate(SWAP_matrix, [i, i + 1], num_gate_qbits))

    identity_padded_gate = identity_pad_gate(gate, [gate_qbits[1] - 1, gate_qbits[1]], num_gate_qbits)

    gates = swap_gates \
            + [identity_padded_gate] \
            + swap_gates[::-1]

    transformed_gate = matmul(gates)

    return identity_pad_gate(transformed_gate, [i for i in range(gate_qbits[0], gate_qbits[1] + 1)], num_qbits)


def to_complex_matrix(matrix: Union[np.ndarray, List]) -> List:
    """
    Convert regular matrix to matrix of ComplexVals.
    :param matrix: any matrix.
    :return: Complex matrix.
    """
    # Return matrix if already complex list.
    if type(matrix) == list and type(matrix[0][0]) == ComplexVal:
        return matrix

    output: List[List] = matrix.tolist()

    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if type(matrix[i, j]) == complex or type(matrix[i, j]) == np.complex128:
                output[i][j] = ComplexVal(matrix[i, j].real, matrix[i, j].imag)
            else:
                output[i][j] = ComplexVal(matrix[i, j])

    return output


def prove_and_get_mean_runtime(function: Callable, num_repeats: int):
    """
    Repeat proof and measure time.
    :param function: circuit.
    :param num_repeats: number of repeats.
    """
    mean_time = 0.0

    for _ in range(num_repeats):
        (_, _, time) = function()
        mean_time += time

    mean_time = mean_time / num_repeats
    print('Mean time: ' + precision_format.format(mean_time) + ' seconds.')


def build_qbit_constraints(qbit_identifiers: List[str],
                           is_reduced_state_space: bool = False,
                           overapproximation: bool = False) -> str:
    """
    Build qbit constraints that restrict qbit values to be located on the Bloch sphere.
    :param qbit_identifiers: qbit identifiers.
    :param is_reduced_state_space: Reduced state space in case of only X, V, and V_dag operations on bitvectors.
    :param overapproximation: False if exact, True for overapproximation (allows all states in the unit cube).
    :return: Qbit constraints in SMT-LIB2 string.
    """
    expr = '\n'

    for identifier in qbit_identifiers:
        if is_reduced_state_space:
            attributes = ['z0', 'z1', 'v0', 'v1']

            conjunctions = []

            for attribute in attributes:
                conjunction = ' '.join(
                    [f'(= {identifier}.{v} {"true" if v == attribute else "false"})' for v in attributes])

                conjunctions.append(f'(and {conjunction})')

            expr += f'(assert (or {" ".join(conjunctions)}))\n'
        else:
            if not overapproximation:
                expr += f';Exact constraints for qbit {identifier}\n'
                expr += f';----------------------\n'

                # phase angles
                expr += f'(declare-fun {identifier}.phi () Real)\n'
                expr += f'(declare-fun {identifier}.theta () Real)\n\n'

                # equality constraints with the amplitudes
                expr += f'(assert (= {identifier}.alpha.r (cos (/ {identifier}.theta 2.0))))\n'
                expr += f'(assert (= {identifier}.alpha.i 0.0))\n'
                expr += f'(assert (= {identifier}.beta.r (* (cos {identifier}.phi) (sin (/ {identifier}.theta 2.0)))))\n'
                expr += f'(assert (= {identifier}.beta.i (* (sin {identifier}.phi) (sin (/ {identifier}.theta 2.0)))))\n\n'

                # angle ranges
                expr += f'(assert (<= 0.0 {identifier}.theta))\n'
                expr += f'(assert (<= {identifier}.theta {pi}))\n'
                expr += f'(assert (<= 0.0 {identifier}.phi))\n'
                expr += f'(assert (< {identifier}.phi (* 2.0 {pi})))\n'

                # zero phi in case of theta = 0 or theta = pi
                expr += f'(assert (=> (= {identifier}.theta 0.0) (= {identifier}.phi 0.0) ))'
                expr += f'(assert (=> (= {identifier}.theta {pi}) (= {identifier}.phi 0.0) ))'
            else:
                expr += f';Overapproximating constraints for qbit {identifier}\n'
                expr += f';----------------------\n'

                expr += f'(assert (<= -1.0 {identifier}.alpha.r))\n'
                expr += f'(assert (<= {identifier}.alpha.r 1.0))\n'

                expr += f'(assert (= {identifier}.alpha.i 0.0))\n'

                expr += f'(assert (<= -1.0 {identifier}.beta.r))\n'
                expr += f'(assert (<= {identifier}.beta.r 1.0))\n'

                expr += f'(assert (<= -1.0 {identifier}.beta.i))\n'
                expr += f'(assert (<= {identifier}.beta.i 1.0))\n'

            expr += '\n\n'

    return expr


def complex_str(real: str, imag: str) -> str:
    """
    String representation of complex number.
    :param real: real part.
    :param imag: imaginary part.
    :return: string representation.
    """
    if '-' in imag:
        imag_abs = imag.split('-')[1]
        return f'{real} - {imag_abs}i'
    else:
        return f'{real} + {imag}i'
