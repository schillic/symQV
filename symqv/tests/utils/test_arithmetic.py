import numpy as np

from symqv.lib.constants import zero, one
from symqv.lib.utils.arithmetic import kron


def test_kron():
    desired = np.zeros((8, 1))
    desired[0, 0] = 1

    assert np.allclose(kron([zero, zero, zero]), desired)

    desired = np.zeros((8, 1))
    desired[1, 0] = 1

    assert np.allclose(kron([zero, zero, one]), desired)

    desired = np.zeros((8, 1))
    desired[2, 0] = 1

    assert np.allclose(kron([zero, one, zero]), desired)

    desired = np.zeros((8, 1))
    desired[3, 0] = 1
    print(desired)

    assert np.allclose(kron([zero, one, one]), desired)


if __name__ == '__main__':
    test_kron()
