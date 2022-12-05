import numpy as np
from z3 import Function, RealSort, Const, BoolSort, IntSort, Real

# functions for the SMT module
from symqv.lib.expressions.complex import ComplexVal

sin = Function("sin", RealSort(), RealSort())
cos = Function("cos", RealSort(), RealSort())
pi = Const("pi", RealSort())
exp = Const("exp", RealSort())

to_int = Function("to_int", BoolSort(), IntSort())
to_bool = Function("to_bool", IntSort(), BoolSort())

# basis states
zero = np.array([[1 + 0j],
                 [0 + 0j]])

one = np.array([[0 + 0j],
                [1 + 0j]])

# gates
I_matrix = np.array([[1, 0],
                     [0, 1]])

X_matrix = np.array([[0, 1],
                     [1, 0]])

Y_matrix = np.array([[0, -1j],
                     [1j, 0]])

Z_matrix = np.array([[1, 0],
                     [0, -1]])

CNOT_matrix = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 1, 0]])

CNOT_reversed_matrix = np.array([[1, 0, 0, 0],
                                 [0, 0, 0, 1],
                                 [0, 0, 1, 0],
                                 [0, 1, 0, 0]])

SWAP_matrix = np.array([[1, 0, 0, 0],
                        [0, 0, 1, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1]])

CZ_matrix = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, -1]])

H_matrix = np.array([[1, 1],
                     [1, -1]]) / np.sqrt(2)

T_matrix = np.array([[1, 0],
                     [0, np.exp(1j * np.pi / 2)]])

S_matrix = np.array([[1, 0],
                     [0, 1j]])

CCX_matrix = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0, 0, 1, 0]])

CCZ_matrix = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0, 0, 0, -1]])

CSWAP_matrix = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1]])

ISWAP_matrix = np.array([[1, 0, 0, 0],
                         [0, 0, 1j, 0],
                         [0, 1j, 0, 0],
                         [0, 0, 0, 1]])

V_matrix = np.array([[1, -1j],
                     [-1j, 1]]) * (1 + 1j) / 2

V_dag_matrix = np.array([[1, 1j],
                         [1j, 1]]) * (1 - 1j) / 2

CV_matrix = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, -1j],
                      [0, 0, -1j, 1]]) * (1 + 1j) / 2

CV_inv_matrix = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, 1j],
                          [0, 0, 1j, 1]]) * (1 - 1j) / 2


def P_matrix(phi: float) -> np.array:
    return np.array([[ComplexVal(1), ComplexVal(0)],
                     [ComplexVal(0), ComplexVal(cos(phi), sin(phi))]])


def Rk_matrix(k: int) -> np.array:
    return np.array([[1, 0],
                     [0, np.exp(2j * np.pi / 2 ** k)]])


def Rx_matrix(theta: float) -> np.array:
    return np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)],
                     [-1j * np.sin(theta / 2), np.cos(theta / 2)]])


def Ry_matrix(theta: float) -> np.array:
    return np.array([[np.cos(theta / 2), -np.sin(theta / 2)],
                     [np.sin(theta / 2), np.cos(theta / 2)]])


def Rz_matrix(theta: float) -> np.array:
    return np.array([[np.exp(-1j * theta / 2), 0],
                     [0, np.exp(1j * theta / 2)]])


def CRk_matrix(k: int) -> np.array:
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, np.exp(2j * np.pi / 2 ** k)]])


def CRZ_matrix(lambd: float) -> np.array:
    return np.array([[1, 0, 0, 0],
                     [0, np.exp(-1j * lambd / 2), 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, np.exp(1j * lambd / 2)]])


def CU1_matrix(lambd: float) -> np.array:
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, np.exp(1j * lambd)]])


def U3_matrix(theta: float, phi: float, lambd: float) -> np.array:
    return np.array([[np.cos(theta / 2), -np.exp(1j * lambd) * np.sin(theta / 2)],
                     [np.exp(1j * phi) * np.sin(theta / 2), np.exp(1j * (phi + lambd)) * np.cos(theta / 2)]])


def U3_symbolic_matrix(theta: Real, phi: Real, lambd: Real) -> np.array:
    return np.array([[cos(theta / 2), -exp(1j * lambd) * sin(theta / 2)],
                     [exp(1j * phi) * sin(theta / 2), exp(1j * (phi + lambd)) * cos(theta / 2)]])
