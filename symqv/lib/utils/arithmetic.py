from typing import List, Tuple, Union

import numpy as np
from z3 import And, Not

from symqv.lib.constants import to_int
from symqv.lib.expressions.complex import ComplexVal
from symqv.lib.expressions.qbit import QbitVal
from symqv.lib.expressions.rqbit import RQbitVal


def qbit_equals_value(qbit: Union[QbitVal, RQbitVal], value: Tuple[Union[int, float], Union[int, float]]):
    """
    SMT qbit equals value.
    :param qbit: qbit.
    :param value: tuple of integer values.
    :return: SMT equals expression.
    """
    if type(value[0]) == int:
        if isinstance(qbit, RQbitVal):
            return And(qbit.z0 == bool(value[0]), qbit.z1 == bool(value[1]))
        return And(qbit.alpha.r == value[0], qbit.alpha.i == 0,
                   qbit.beta.r == value[1], qbit.beta.i == 0)
    else:
        return And(qbit.alpha.r == value[0].real, qbit.alpha.i == value[0].imag,
                   qbit.beta.r == value[1].real, qbit.beta.i == value[1].imag)


def rqbit_equals_rqbit(rqbit_a: RQbitVal, rqbit_b: RQbitVal):
    """
    SMT RQbit equals other RQbit.
    :param rqbit_a: first.
    :param rqbit_b: second.
    :return: SMT equals.
    """
    return rqbit_a.z0 == rqbit_b.z0 and rqbit_a.z1 == rqbit_b.z1 and \
           rqbit_a.h0 == rqbit_b.h0 and rqbit_a.h1 == rqbit_b.h1 and \
           rqbit_a.zm0 == rqbit_b.zm0 and rqbit_a.zm1 == rqbit_b.zm1 and \
           rqbit_a.hm0 == rqbit_b.hm0 and rqbit_a.hm1 == rqbit_b.hm1 and \
           rqbit_a.v0 == rqbit_b.v0 and rqbit_a.v1 == rqbit_b.v1


def qbit_isclose_to_value(qbit: QbitVal, value: Tuple[Union[int, float], Union[int, float]],
                          delta: float = 0.0001):
    """
    SMT qbit is close to value.
    :param qbit: qbit.
    :param value: tuple of integer values.
    :param delta: error tolerance (absolute).
    :return: SMT equals expression.
    """
    if type(value[0]) == int:
        return And(value[0] - delta <= qbit.alpha.r, qbit.alpha.r <= value[0] + delta,
                   -delta <= qbit.alpha.i, qbit.alpha.i <= delta,
                   value[1] - delta <= qbit.beta.r, qbit.beta.r <= value[1] + delta,
                   -delta <= qbit.beta.i, qbit.beta.i <= delta)
    else:
        return And(value[0].real - delta <= qbit.alpha.r, qbit.alpha.r <= value[0].real + delta,
                   value[0].imag - delta <= qbit.alpha.i, qbit.alpha.i <= value[0].imag + delta,
                   value[1].real - delta <= qbit.beta.r, qbit.beta.r <= value[1].real + delta,
                   value[1].imag - delta <= qbit.beta.i, qbit.beta.i <= value[1].imag + delta)


def qbits_equal(qbit_a: QbitVal, qbit_b: QbitVal):
    """
    SMT qbit equals qbit.
    :param qbit_a: first.
    :param qbit_b: second.
    :return: SMT equals expression.
    """
    return And(qbit_a.alpha.r == qbit_b.alpha.r, qbit_a.alpha.i == qbit_b.alpha.i,
               qbit_a.beta.r == qbit_b.beta.r, qbit_a.beta.i == qbit_b.beta.i)


def state_equals(psi: List, psi_prime: List):
    if len(psi) != len(psi_prime):
        raise Exception(
            f'States are not the same dimension, first is dimension {len(psi)}, second is dimension {len(psi_prime)}.')

    elements = []

    for i in range(len(psi)):
        elements.append(psi_prime[i] == psi[i])

    return And(elements)


def state_equals_phase_oracle(psi: List, psi_prime: List, oracle_value: int):
    if len(psi) != len(psi_prime):
        raise Exception(
            f'States are not the same dimension, first is dimension {len(psi)}, second is dimension {len(psi_prime)}.')

    if oracle_value > len(psi) - 1:
        raise Exception(f'Oracle value {oracle_value} is not in the value range 0 to {len(psi)}.')

    elements = []

    for i in range(len(psi)):
        if i == oracle_value:
            elements.append(psi_prime[i] == psi[i] * (-1))
        else:
            elements.append(psi_prime[i] == psi[i])

    return And(elements)


def state_not_equals(psi: List, psi_prime: List):
    if len(psi) != len(psi_prime):
        raise Exception("States are not the same dimension.")

    elements = []

    for i in range(len(psi)):
        elements.append(psi[i] == psi_prime[i])

    return Not(And(elements))


def qbit_kron(qbit_a: QbitVal, qbit_b: QbitVal) -> List:
    """
    Kronecker product of two qbits.
    :param qbit_a: first qbit.
    :param qbit_b: second qbit.
    :return: Kronecker product.
    """
    return complex_kron(qbit_a.to_complex_list(), qbit_b.to_complex_list())


def qbit_kron_n_ary(qbits: List[QbitVal]) -> List:
    """
    Kronecker product of n qbits.
    :param qbits: list of qbits.
    :return: N-ary kronecker product.
    """
    kronecker_product = None

    for qbit in qbits:
        if kronecker_product is None:
            kronecker_product = qbit.to_complex_list()
        else:
            kronecker_product = complex_kron(kronecker_product, qbit.to_complex_list())

    return kronecker_product


def complex_kron(vector_a: List[ComplexVal], vector_b: List[ComplexVal]) -> List:
    """
    Kronecker product of two complex vectors.
    :param vector_a: first vector.
    :param vector_b: second vector.
    :return: Kronecker product.
    """
    output_vector = [None] * (len(vector_a) * len(vector_b))

    for i, entry_1 in enumerate(vector_a):
        for j, entry_2 in enumerate(vector_b):
            output_vector[(len(vector_b)) * i + j] = entry_1 * entry_2

    return output_vector


def complex_kron_n_ary(vectors: List[List[ComplexVal]]) -> List:
    """
    N-ary Kronecker product of >= 2 complex vectors.
    :param vectors: list of vectors.
    :return: N-ary Kronecker product.
    """
    if len(vectors) == 0:  # or not isinstance(vectors[0][0], ComplexVal):
        raise Exception('Illegal argument: needs to be a list of at least 2 complex vectors.')

    if isinstance(vectors[0][0], int):
        for i in range(len(vectors)):
            for j in range(len(vectors[i])):
                vectors[i][j] = to_int(vectors[i][j])

    kronecker_product = None

    for vector in vectors:
        if kronecker_product is None:
            kronecker_product = vector
        else:
            kronecker_product = complex_kron(kronecker_product, vector)

    return kronecker_product


def complex_matrix_kron(matrix_a: np.ndarray, matrix_b: np.ndarray):
    """
    Kronecker product of two complex matrices.
    :param matrix_a: first matrix.
    :param matrix_b: second matrix.
    :return: Kronecker product.
    """

    def cast(num):
        if type(num) == np.int64:
            return ComplexVal(int(num))
        return num

    return [[cast(num1) * cast(num2)
             for num1 in elem1
             for num2 in matrix_b[row]]
            for elem1 in matrix_a
            for row in range(len(matrix_b))]


def complex_matrix_kron_n_ary(matrices: List[np.ndarray]) -> List:
    """
    N-ary Kronecker product of >= 2 complex matrices.
    :param matrices: list of matrices.
    :return: N-ary Kronecker product.
    """
    kronecker_product = None

    for matrix in matrices:
        if kronecker_product is None:
            kronecker_product = matrix
        else:
            kronecker_product = complex_matrix_kron(kronecker_product, matrix)

    return kronecker_product


def kron(vectors: List[np.ndarray]) -> np.ndarray:
    """
    Numpy-based n-ary Kronecker product.
    :param vectors: list of vectors or matrices.
    :return: N-ary Kronecker product.
    """
    kronecker_product = None

    # Special case: symbolic matrix entries
    if np.dtype('O') in [entry.dtype for entry in vectors]:
        kronecker_product = complex_matrix_kron_n_ary(vectors)
    else:
        # Standard case: concrete matrix entries
        for vector in vectors:
            if kronecker_product is None:
                kronecker_product = vector
            else:
                kronecker_product = np.kron(kronecker_product, vector)

    return kronecker_product


def matmul(matrices: List[np.ndarray]) -> np.ndarray:
    """
    Numpy n-ary matrix product.
    :param matrices: list of matrices.
    :return: N-ary Matrix product.
    """
    product = None

    for matrix in matrices:
        if product is None:
            product = matrix
        else:
            product = np.matmul(matrix, product)

    return product


def matrix_vector_multiplication(matrix: List[List], vector: List) -> List:
    if len(matrix[0]) != len(vector):
        raise Exception(f'Matrix column count ({len(matrix[0])}) has to be equal to vector row count ({len(vector)}).')

    # General case
    m = len(matrix[0])
    n = len(matrix)

    output_vector = [ComplexVal(0)] * m

    for i in range(m):
        for k in range(n):
            if isinstance(matrix[i][k], ComplexVal) and matrix[i][k].r == 0 and matrix[i][k].i == 0:
                continue
            else:
                output_vector[i] += matrix[i][k] * vector[k]

    return output_vector
