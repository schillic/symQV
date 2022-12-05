from typing import List

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
from qiskit.quantum_info import DensityMatrix, partial_trace
from scipy import optimize
from tqdm import tqdm

from quavl.lib.constants import zero
from quavl.lib.operations.state_decomposition import separate_kth_qbit_from_state, from_angles, euclidean_distance
from quavl.lib.parsing.open_qasm import read_from_qasm
from quavl.lib.utils.arithmetic import kron

bounds = [(0, 2 * np.pi), (0, np.pi)]

path = 'benchmarks/quavl/teleportv4.qasm'


def fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    r"""
    Given two density operators ρ and σ, the fidelity is generally defined as the quantity
    F(\rho, \sigma)=\left(\operatorname{tr} {\sqrt {{\sqrt {\rho }}\sigma {\sqrt {\rho }}}}\right)^{2}.
    :param rho: First density matrix ρ.
    :param sigma: Second density matrix σ.
    :return: Fidelity.
    """
    if not isinstance(rho, np.ndarray) or not isinstance(sigma, np.ndarray):
        raise ValueError("Arguments rho and sigma must be of type np.ndarray. "
                         f"Received type={type(rho)} for rho and type={type(sigma)} for sigma.")

    if rho.shape != sigma.shape:
        raise ValueError("The dimensions of both arguments must be equal.")

    is_square_matrix = rho.shape[0] == rho.shape[1]

    if not is_square_matrix and rho.shape[1] != 1:
        raise ValueError("Both arguments most be either square matrices or column vectors. "
                         f"Received shape={rho.shape} for rho and shape={sigma.shape} for sigma.")

    if is_square_matrix:
        fidelity_value = np.trace(np.sqrt(np.sqrt(rho) * sigma * np.sqrt(rho)))
    else:
        fidelity_value = scipy.linalg.norm(np.dot(rho.T, sigma)) ** 2

    if fidelity_value.imag != 0:
        raise ValueError(f"Fidelity should have no imaginary component, but was {fidelity_value}")

    return float(fidelity_value.real)


def objective_function_teleportation(x: List[float]):
    """
    Objective function for verifying teleportation.
    It is zero exactly when the desired output is matched.
    :param x: phi and theta.
    :return: objective value (euclidean distance between desired and actual output).
    """
    phi = x[0]
    theta = x[1]

    psi = from_angles(phi, theta)
    initial_state = kron([zero, zero, psi])

    qc = read_from_qasm(path)
    final_state = qc.get_final_states(initial_state)[0]

    output_qbit, separable, result = separate_kth_qbit_from_state(final_state, 0)

    if not separable:
        raise Exception('Entangled qbit.\n' + str(result))

    # dist = euclidean_distance(output_qbit, psi)
    dist = fidelity(output_qbit, psi)

    if dist > 1e-14:
        pass
        # print(f'dist = {dist} \t psi = {psi.flatten()}\t output = {output_qbit.flatten()}')

    return -dist


def verify_teleportation():
    # if result is non-zero, we found a counter-example
    result = optimize.differential_evolution(objective_function_teleportation, bounds)

    print('\nVerification result:')
    print(result)

    phi, theta = result.x[0], result.x[1]

    print(f"Objective minimum = {'{0:.4f}'.format(float(result.fun))}")
    print(f"Min arg: phi = {'{0:.4f}'.format(phi)}, theta = {'{0:.4f}'.format(theta)}")
    cartesian = from_angles(phi, theta)

    alpha = cartesian[0, 0]
    beta = cartesian[1, 0]
    print(f"alpha = {'{0:.4f}'.format(alpha.real)} + {'{0:.4f}'.format(alpha.imag)}j")
    print(f"beta = {'{0:.4f}'.format(beta.real)} + {'{0:.4f}'.format(beta.imag)}j")

    psi = from_angles(phi, theta)
    initial_state = kron([zero, zero, psi])

    qc = read_from_qasm(path)
    final_state = qc.get_final_states(initial_state)[0]

    output_qbit, separable, result = separate_kth_qbit_from_state(final_state, 0)

    print('Actual quantum circuit output:')
    print(output_qbit.flatten())


def test_teleportation():
    psi = from_angles(0.3, 0.4)
    initial_state = kron([zero, zero, psi])
    print(initial_state)

    qc = read_from_qasm(path)
    final_state = qc.get_final_states(initial_state)[0]
    print(final_state)

    trace_2 = partial_trace(final_state, [2])
    print(trace_2)

    # print(separate_kth_qbit_from_state(final_state, 2))


def visualize_error():
    phi = np.arange(0, 2 * np.pi, 0.05)
    theta = np.arange(0, np.pi, 0.05)

    Phi, Theta = np.meshgrid(phi, theta)

    # evaluation of the function on the grid
    error = np.array([objective_function_teleportation([Phi[i, j], Theta[i, j]])
                      for i in tqdm(range(len(Phi))) for j in range(len(Phi[0]))])

    error = error.reshape((len(Phi), len(Phi[0])))

    # drawing the function
    print(f"Error: min = {np.min(error)}, Max = {np.max(error)}")
    im = plt.imshow(error,
                    cmap=plt.cm.rainbow,
                    vmax=np.sqrt(2),
                    extent=[0, 2 * np.pi, 0, np.pi])

    # adding the color bar on the right
    plt.colorbar(im)

    # latex title
    plt.xlabel("$\\phi$")
    plt.ylabel("$\\theta$")
    plt.xticks(np.linspace(0, 2 * np.pi, 5),
               ['0', "$\\pi/2$", "$\\pi$", "$3\\pi/2$", "$2\\pi$"])
    plt.yticks(np.linspace(0, np.pi, 3),
               ['0', "$\\pi/2$", "$\\pi$"])
    plt.title('error')
    plt.show()


if __name__ == "__main__":
    test_teleportation()
