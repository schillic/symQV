import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

from symqv.lib.constants import zero, one
from symqv.lib.operations.state_decomposition import swap_kth_qbit_to_front, from_angles, build_objective_function, \
    separate_first_qbit_from_state, to_angles, trace_distance, fidelity, \
    c_metric, euclidean_distance, separate_kth_qbit_from_state
from symqv.lib.utils.arithmetic import kron


def test_euclidean_distance():
    q1 = from_angles(0.3, 0.7)
    q2 = from_angles(0.3, 0.7)

    assert euclidean_distance(q1, q2) == 0
    assert euclidean_distance(zero, one) == np.sqrt(2)

    psi = np.array([[0.98472654 + 0j],
                    [-0.10008153 - 0.1424687j]])

    assert euclidean_distance(psi, psi) == 0


def test_trace_distance():
    q1 = from_angles(0.3, 0.7)
    q2 = from_angles(0.3, 0.7)

    assert trace_distance(q1, q2) == 0

    assert trace_distance(one, zero) == 0


def test_fidelity():
    q1 = from_angles(0.3, 0.7)
    q2 = from_angles(0.3, 0.7)

    assert np.isclose(fidelity(q1, q2), 1)

    assert fidelity(one, zero) == 0


def test_c_metric():
    q1 = from_angles(0.3, 0.7)
    q2 = from_angles(0.3, 0.7)

    assert np.isclose(c_metric(q1, q2), 0, atol=1.e-7)

    assert np.isclose(c_metric(one, zero), 1)


def test_build_objective_function():
    state = kron([one, zero, zero])

    objective_function = build_objective_function(state)

    assert objective_function(to_angles(one)) == 1.0
    assert objective_function(to_angles(zero)) == 0

    phi, theta = 0.3, 0.7

    qbit = from_angles(phi, theta)
    state = kron([qbit, zero, zero])

    objective_function = build_objective_function(state)

    assert objective_function([phi, theta]) == 0
    assert np.isclose(objective_function([phi + 0.1, theta]), 0.099958)
    assert np.isclose(objective_function([phi - 0.1, theta]), 0.099958)

    assert np.isclose(objective_function([phi, theta + 0.1]), 0.139343)
    assert np.isclose(objective_function([phi, theta - 0.1]), 0.177029)

    assert objective_function([phi + np.pi, theta]) == 2


def visualize_build_objective_function():
    def visualize_loss(qbit):
        state = kron([qbit, zero, zero])

        objective_function = build_objective_function(state)

        phi = np.arange(0, 2 * np.pi, 0.05)
        theta = np.arange(0, np.pi, 0.05)

        Phi, Theta = np.meshgrid(phi, theta)

        # evaluation of the function on the grid
        error = np.array([objective_function([Phi[i, j], Theta[i, j]])
                          for i in range(len(Phi)) for j in range(len(Phi[0]))])
        error = error.reshape((len(Phi), len(Phi[0])))

        # minimum value
        min_arg = optimize.shgo(objective_function, [(0, 2 * np.pi), (0, np.pi)], iters=10)

        # drawing the function
        print(np.max(error))
        im = plt.imshow(error,
                        cmap=plt.cm.rainbow,
                        # norm=colors.LogNorm(vmin=0.000001, vmax=np.max(error)),
                        extent=[0, 2 * np.pi, 0, np.pi])

        # adding the color bar on the right
        plt.colorbar(im)

        print(min_arg)
        plt.scatter(min_arg.x[0], min_arg.x[1],
                    marker='o',
                    facecolors='None',
                    edgecolors='white')

        # latex title
        plt.xlabel("$\\phi$")
        plt.ylabel("$\\theta$")
        plt.xticks(np.linspace(0, 2 * np.pi, 5),
                   ['0', "$\\pi/2$", "$\\pi$", "$3\\pi/2$", "$2\\pi$"])
        plt.yticks(np.linspace(0, np.pi, 3),
                   ['0', "$\\pi/2$", "$\\pi$"])
        plt.title('error')
        plt.show()

    qbit = np.array([[6.2831853], [0.05]])
    visualize_loss(qbit)

    qbit = one
    visualize_loss(qbit)

    qbit = from_angles(0.3, 0.7)
    visualize_loss(qbit)

    qbit = from_angles(np.pi, np.pi / 2)
    visualize_loss(qbit)


def test_separate_first_qbit_from_state():
    state = kron([one, zero, zero])
    x, separable, res = separate_first_qbit_from_state(state)

    assert np.allclose(x, one)

    state = kron([zero, zero, zero])
    x, separable, res = separate_first_qbit_from_state(state)

    assert np.allclose(x, zero)

    psi = from_angles(0.3, 0.4)
    state = kron([psi, zero, zero])
    x, separable, res = separate_first_qbit_from_state(state)

    assert np.allclose(x, psi)

    state = kron([psi, psi, zero])
    x, separable, res = separate_first_qbit_from_state(state)

    assert np.allclose(x, psi)


def test_separate_kth_qbit_from_state():
    state = kron([zero, zero, one])
    x, separable, res = separate_kth_qbit_from_state(state, 2)

    assert np.allclose(x, one)

    psi = from_angles(0.3, 0.4)
    state = kron([zero, zero, psi])
    x, separable, res = separate_kth_qbit_from_state(state, 2)
    assert np.allclose(x, psi)

    state = kron([zero, psi, psi])
    x, separable, res = separate_kth_qbit_from_state(state, 2)

    assert np.allclose(x, psi)


def test_swap_kth_qbit_to_front():
    state = kron([zero, zero, one])
    result = swap_kth_qbit_to_front(state, 2)

    desired = kron([one, zero, zero])

    assert np.array_equal(result, desired)


if __name__ == "__main__":
    test_separate_kth_qbit_from_state()
