from typing import Tuple, List, Callable, Optional

import numpy as np
from scipy import optimize
from scipy.optimize import OptimizeResult

from symqv.lib.constants import SWAP_matrix, zero, one
from symqv.lib.utils.helpers import identity_pad_gate


def to_angles(qbit: np.array) -> np.array:
    alpha = qbit[0]
    beta = qbit[1]

    def magnitude(z: np.complex64) -> float:
        return np.sqrt(z.real ** 2 + z.imag ** 2)

    theta = 2 * np.arccos(magnitude(alpha))

    if magnitude(alpha) == 0 or magnitude(beta) == 0:
        phi = 0
    else:
        phi = np.angle(beta) - np.angle(alpha)

    return np.array([phi, theta], dtype=float)


def from_angles(phi: float, theta: float) -> np.array:
    """
    Generate a qbit from two angular values.
    :param phi: first angle.
    :param theta: second angle.
    :return: qbit vector.
    """
    if not np.isscalar(phi):
        raise ValueError(f'phi was not a scalar: shape {phi.shape}')

    if not np.isscalar(theta):
        raise ValueError(f'theta was not a scalar: shape {theta.shape}')

    if theta < 0 or theta > np.pi or phi < 0 or phi > 2 * np.pi:
        raise ValueError(f'Range for (phi, theta) is ([0, 2pi), [0, pi]), but' + \
                         (f' phi = {phi} >= {2 * np.pi}' if phi >= 2 * np.pi else '') + \
                         (f' theta = {theta} > {np.pi}' if theta > np.pi else ''))

    return np.array([[np.cos(theta / 2)],
                     [np.exp(1j * phi) * np.sin(theta / 2)]])


def euclidean_distance(first: np.array, second: np.array) -> float:
    """
    Euclidean distance between two complex vectors.
    :param first: first vector.
    :param second: second vector.
    :return: Euclidean distance.
    """
    return np.linalg.norm(first - second)


def to_density_matrix(vector: np.array) -> np.ndarray:
    """
    Convert quantum state vector to a density matrix.
    :param vector: column vector representing a quantum state.
    :return: density matrix.
    """
    return np.matmul(vector, vector.T.conj())


def trace_distance(first: np.array, second: np.array) -> float:
    """
    The trace distance between density matrices ρ and σ is defined by D(ρ, σ) ≡ 1/2 tr|ρ−σ|.
    :param first: first state vector.
    :param second: second state vector.
    :return: Trace distance.
    """
    # vectors to density matrices
    density_first = to_density_matrix(first)
    density_second = to_density_matrix(second)

    return 0.5 * np.trace(density_first - density_second)


def fidelity(first: np.array, second: np.array) -> float:
    """
    Fidelity between density matrices ρ and σ is defined by D(ρ, σ) ≡ 1/2 tr|ρ−σ|.
    0 ≤ F(ρ, σ) ≤ 1, with equality in the second inequality if and only if ρ = σ.
    :param first: first state vector.
    :param second: second state vector.
    :return: Fidelity.
    """
    # vectors to density matrices
    density_second = to_density_matrix(second)

    return np.matmul(np.matmul(first.T.conj(), density_second), first)[0, 0]


def c_metric(first: np.array, second: np.array) -> float:
    """
    C metric between density matrices ρ and σ is defined C(ρ,σ) ≡ sqrt(1−F(ρ,σ)).

    From: Distance measures to compare real and ideal quantum processes (2009) by
    Alexei Gilchrist, Nathan K. Langford, and Michael A. Nielsen.
    :param first: first state vector.
    :param second: second state vector.
    :return: Bures metric.
    """
    return np.sqrt(1 - fidelity(first, second))


def swap_kth_qbit_to_front(state: np.array, k: int) -> np.array:
    """
    Swap kth qbit to the front.
    :param state: quantum state vector.
    :param k: qbit index of qbit that should be swapped to front.
    :return: state with qbit k swapped to the front.
    """
    n = int(np.log2(len(state)))

    if k == 0:
        return state
    if k >= len(state):
        raise ValueError(f'State has {n} qbits, but k was {k}.')

    swapped_state = state

    # looping backwards
    for i in range(k - 1, -1, -1):
        padded_op = identity_pad_gate(SWAP_matrix, [i, i + 1], n)
        swapped_state = np.matmul(padded_op, swapped_state)

    return swapped_state


def separate_kth_qbit_from_state(state: np.array, k: int) -> Tuple[np.array, bool, OptimizeResult]:
    """
    Separate the kth qbit in the quantum state vector and return it.
    :param state: state vector.
    :param k: qbit index of qbit that should be returned.
    :return: single qbit vector and True if separable, False if entangled.
    """
    swapped_state = swap_kth_qbit_to_front(state, k)
    return separate_first_qbit_from_state(swapped_state)


def separate_first_qbit_from_state(state: np.array) -> Tuple[np.array, bool, Optional[OptimizeResult]]:
    """
    Separate the first qbit in the quantum state vector and return it.
    :param state: state vector.
    :return: single qbit vector and True if separable, False if entangled.
    """
    bounds = [(0, 2 * np.pi), (0, np.pi)]

    n = state.shape[0]
    mid = n // 2

    first_half = state[0:mid, :]
    second_half = state[mid:n, :]

    # basis state ket 0
    if np.all(second_half == 0):
        return zero, True, None

    # basis state ket 1
    if np.all(first_half == 0):
        return one, True, None

    # other qbits in basis state:
    if np.count_nonzero(state) == 2:
        return np.array([state[state != 0]]).T, True, None

    # no basis states involved
    objective_function = build_objective_function(state)

    result = optimize.shgo(objective_function, bounds, iters=10)  # , options={'f_min': 0})
    result['quantum_state_vector'] = state

    min_arg = result.x
    min = objective_function(min_arg)

    return from_angles(min_arg[0], min_arg[1]), \
           bool(np.isclose(min.real, 0) and np.isclose(min.imag, 0)), \
           result


def build_objective_function(state: np.array) -> Callable:
    """
    Build an objective function that goes to an optimizer.
    It is zero exactly when its arguments are the angular coordinates of the first qbit in the state vector.
    :param state: quantum state.
    :return: an objective function that returns an objective value (euclidean distance).
    """
    n = state.shape[0]
    mid = n // 2

    first_half = state[0:mid, :]
    second_half = state[mid:n, :]

    def objective_function(x: List[float]) -> float:
        """
        Objective function that is zero exactly when its arguments are the angular coordinates
        of the first qbit in the state vector.
        :return: objective value (euclidean distance).
        """
        phi = x[0]
        theta = x[1]

        cartesian = from_angles(phi, theta)
        alpha = cartesian[0, 0]
        beta = cartesian[1, 0]

        first_by_alpha = (first_half / alpha) if alpha != 0 else np.zeros_like(first_half)
        second_by_beta = (second_half / beta) if beta != 0 else np.zeros_like(second_half)

        return euclidean_distance(first_by_alpha, second_by_beta)

    return objective_function
