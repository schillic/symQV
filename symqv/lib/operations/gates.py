from typing import List, Union

import numpy as np
from z3 import Sqrt, Real

from symqv.lib.constants import X_matrix, H_matrix, CNOT_matrix, CZ_matrix, SWAP_matrix, CNOT_reversed_matrix, \
    Y_matrix, \
    Z_matrix, T_matrix, S_matrix, CCX_matrix, CSWAP_matrix, ISWAP_matrix, I_matrix, CCZ_matrix, CV_matrix, \
    CV_inv_matrix, Rk_matrix, V_matrix, V_dag_matrix, U3_matrix, Rx_matrix, Ry_matrix, Rz_matrix, P_matrix
from symqv.lib.expressions.complex import _to_complex
from symqv.lib.expressions.qbit import QbitVal
from symqv.lib.expressions.rqbit import RQbitVal
from symqv.lib.models.gate import Gate


def I(qbit: QbitVal) -> Gate:
    """
    Identity.
    :param qbit: parameter.
    :return: Gate.
    """
    return Gate('I',
                [qbit],
                I_matrix,
                mapping=lambda q: q)


def X(qbit: Union[QbitVal, RQbitVal]) -> Gate:
    """
    Pauli X (NOT) gate.
    :param qbit: parameter.
    :return: Gate.
    """
    return Gate('X',
                [qbit],
                X_matrix,
                mapping=lambda q: QbitVal(q.beta,
                                          q.alpha),
                r_mapping=lambda rq: rq.__neg__())


def Peres(qbit_1: QbitVal, qbit_2: QbitVal, qbit_3: QbitVal) -> Gate:
    """
    Peres gate.
    :param qbit_1: first parameter.
    :param qbit_2: second parameter.
    :param qbit_3: third parameter.
    :return: Gate.
    """
    return Gate('Peres',
                [qbit_1, qbit_2, qbit_3],
                matrix=np.matmul(np.kron(CNOT_matrix, I_matrix), CCX_matrix))


def Peres_inv(qbit_1: QbitVal, qbit_2: QbitVal, qbit_3: QbitVal) -> Gate:
    """
    Peres gate.
    :param qbit_1: first parameter.
    :param qbit_2: second parameter.
    :param qbit_3: third parameter.
    :return: Gate.
    """
    return Gate('Peres_inv',
                [qbit_1, qbit_2, qbit_3],
                matrix=np.matmul(CCX_matrix, np.kron(CNOT_matrix, I_matrix)))


def Y(qbit: QbitVal) -> Gate:
    """
    Pauli Y gate.
    :param qbit: parameter.
    :return: Gate.
    """
    return Gate('Y',
                [qbit],
                Y_matrix)


def Z(qbit: QbitVal) -> Gate:
    """
    Pauli Z (phase-flip) gate.
    :param qbit: parameter.
    :return: Gate.
    """
    return Gate('Z',
                [qbit],
                Z_matrix,
                mapping=lambda q: QbitVal(q.alpha,
                                          q.beta * (-1)))


def H(qbit: QbitVal) -> Gate:
    """
    Hadamard gate.
    :param qbit: parameter.
    :return: Gate.
    """
    root2 = 1 / Sqrt(Real(2))

    return Gate('H',
                [qbit],
                H_matrix,
                mapping=lambda q: QbitVal((q.alpha + q.beta) * root2,
                                          (q.alpha - q.beta) * root2))


def CNOT(qbit_1: QbitVal, qbit_2: QbitVal) -> Gate:
    """
    Conditional NOT gate.
    :param qbit_1: first parameter.
    :param qbit_2: second parameter.
    :return: Gate.
    """
    return Gate('CNOT',
                [qbit_1, qbit_2],
                CNOT_matrix,
                CNOT_reversed_matrix)


def SWAP(qbit_1: QbitVal, qbit_2: QbitVal) -> Gate:
    """
    SWAP gate.
    :param qbit_1: first parameter.
    :param qbit_2: second parameter.
    :return: Gate.
    """
    return Gate('SWAP',
                [qbit_1, qbit_2],
                SWAP_matrix,
                mapping=lambda q1, q2: (q2, q1))


def CZ(qbit_1: QbitVal, qbit_2: QbitVal) -> Gate:
    """
    Conditional Z gate.
    :param qbit_1: first parameter.
    :param qbit_2: second parameter.
    :return: Gate.
    """
    return Gate('CZ',
                [qbit_1, qbit_2],
                CZ_matrix)


def T(qbit: QbitVal) -> Gate:
    """
    pi/4 phase shift gate.
    :param qbit: parameter.
    :return: Gate.
    """
    return Gate('T',
                [qbit],
                T_matrix)


def S(qbit: QbitVal) -> Gate:
    """
    pi/2 phase shift gate.
    :param qbit: parameter.
    :return: Gate.
    """
    return Gate('S',
                [qbit],
                S_matrix)


def oracle(qbits: List[QbitVal], oracle_value: int) -> Gate:
    """
    Oracle modeled as gate.
    :param qbits: parameters.
    :param oracle_value: value for the oracle to phase flip.
    :return: Gate.
    """
    return Gate('oracle',
                qbits,
                None,
                oracle_value=oracle_value)


def CCX(qbit_1: QbitVal, qbit_2: QbitVal, qbit_3: QbitVal) -> Gate:
    """
    Toffoli (CCNOT) gate.
    :param qbit_1: first parameter.
    :param qbit_2: second parameter.
    :param qbit_3: third parameter.
    :return: Gate.
    """
    return Gate('CCX',
                [qbit_1, qbit_2, qbit_3],
                CCX_matrix)


def CCZ(qbit_1: QbitVal, qbit_2: QbitVal, qbit_3: QbitVal) -> Gate:
    """
    Conditional CZ gate.
    :param qbit_1: first parameter.
    :param qbit_2: second parameter.
    :param qbit_3: third parameter.
    :return: Gate.
    """
    return Gate('CCZ',
                [qbit_1, qbit_2, qbit_3],
                CCZ_matrix)


def CSWAP(qbit_1: QbitVal, qbit_2: QbitVal, qbit_3: QbitVal) -> Gate:
    """
    Conditional SWAP (Fredkin) gate.
    :param qbit_1: first parameter.
    :param qbit_2: second parameter.
    :param qbit_3: third parameter.
    :return: Gate.
    """
    return Gate('CSWAP',
                [qbit_1, qbit_2, qbit_3],
                CSWAP_matrix)


def V(qbit: QbitVal) -> Gate:
    """
    V gate.
    :param qbit: parameter.
    :return: Gate.
    """
    return Gate('V',
                [qbit],
                V_matrix,
                r_mapping=lambda rq: RQbitVal(v0=rq.z0, v1=rq.z1, z0=rq.v1, z1=rq.v0))


def V_dag(qbit: QbitVal) -> Gate:
    """
    V+ gate.
    :param qbit: parameter.
    :return: Gate.
    """
    return Gate('V+',
                [qbit],
                V_dag_matrix,
                r_mapping=lambda rq: RQbitVal(v0=rq.z1, v1=rq.z0, z0=rq.v0, z1=rq.v1))


def CV(qbit_1: QbitVal, qbit_2: QbitVal) -> Gate:
    """
    Conditional V gate.
    :param qbit_1: first parameter.
    :param qbit_2: second parameter.
    :return: Gate.
    """
    return Gate('CV',
                [qbit_1, qbit_2],
                CV_matrix)


def CV_inv(qbit_1: QbitVal, qbit_2: QbitVal) -> Gate:
    """
    Conditional V+ gate.
    :param qbit_1: first parameter.
    :param qbit_2: second parameter.
    :return: Gate.
    """
    return Gate('CV+',
                [qbit_1, qbit_2],
                CV_inv_matrix)


def Rx(qbit: QbitVal, angle: float) -> Gate:
    """
    X-Rotation gate.
    :param qbit: parameter.
    :param angle: angle.
    :return: Gate
    """
    return Gate('Rx',
                [qbit],
                Rx_matrix(angle))


def Ry(qbit: QbitVal, angle: float) -> Gate:
    """
    Y-Rotation gate.
    :param qbit: parameter.
    :param angle: angle.
    :return: Gate
    """
    return Gate('Ry',
                [qbit],
                Ry_matrix(angle))


def Rz(qbit: QbitVal, angle: float) -> Gate:
    """
    Z-Rotation gate.
    :param qbit: parameter.
    :param angle: angle.
    :return: Gate
    """
    return Gate('Rz',
                [qbit],
                Rz_matrix(angle))


def P(qbit: QbitVal, phi: float, use_phase_transform: bool = True) -> Gate:
    """
    Parametrized phase gate.
    :param qbit: parameter.
    :param phi: rotation parameter.
    :param use_phase_transform: realize transformation on the phase values instead of the amplitudes.
    :return: Gate.
    """
    mapping = lambda q: QbitVal(alpha=q.alpha,
                                beta=_to_complex(np.exp(1j * phi)) * q.beta)

    if use_phase_transform:
        mapping = lambda q: QbitVal(phi=q.phi,
                                    theta=q.theta + phi)

    return Gate('P',
                [qbit],
                P_matrix(phi),
                parameter=phi,
                mapping=mapping)


def R(qbit: QbitVal, k: int, use_phase_transform: bool = False) -> Gate:
    """
    Parametrized phase gate.
    :param qbit: parameter.
    :param k: rotation parameter.
    :param use_phase_transform: realize transformation on the phase values instead of the amplitudes.
    :return: Gate.
    """
    mapping = lambda q: QbitVal(alpha=q.alpha,
                                beta=_to_complex(np.exp(2j * np.pi / 2 ** k)) * q.beta)

    if use_phase_transform:
        mapping = lambda q: QbitVal(phi=q.phi + 2 * np.pi / 2 ** k,
                                    theta=q.theta)

    return Gate('R',
                [qbit],
                Rk_matrix(k),
                parameter=k,
                mapping=mapping)


def ISWAP(qbit_1: QbitVal, qbit_2: QbitVal) -> Gate:
    """
    iSWAP gate.
    :param qbit_1: first parameter.
    :param qbit_2: second parameter.
    :return: Gate.
    """
    return Gate('ISWAP',
                [qbit_1, qbit_2],
                ISWAP_matrix)


def U3(qbit: QbitVal, theta: float, phi: float, lambd: float) -> Gate:
    """
    Single-qubit rotation gate with 3 Euler angles.
    :param qbit: parameter.
    :param theta: first angle.
    :param phi: second angle.
    :param lambd: third angle.
    :return:
    """
    return Gate('U3',
                [qbit],
                U3_matrix(theta, phi, lambd))


def custom_gate(name: str, qbits: List[QbitVal], transformation_matrix: np.ndarray):
    """
    Custom gate.
    :param name: Custom gate name.
    :param qbits: List of parameters.
    :param transformation_matrix: unitary matrix.
    :return: Gate.
    """
    if transformation_matrix.shape[0] != 2 ** len(qbits):
        raise Exception('Wrong matrix dimension')

    if (np.matmul(transformation_matrix, transformation_matrix) != np.identity(transformation_matrix.shape[0])).all():
        raise Exception('Matrix must be unitary')

    return Gate(name,
                qbits,
                transformation_matrix)
