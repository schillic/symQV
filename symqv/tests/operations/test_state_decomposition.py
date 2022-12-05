import numpy as np

from symqv.lib.constants import zero, one
from symqv.lib.operations.state_decomposition import swap_kth_qbit_to_front, from_angles, \
    separate_first_qbit_from_state, separate_kth_qbit_from_state
from symqv.lib.utils.arithmetic import kron


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
