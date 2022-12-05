import numpy as np

from symqv.lib.utils.state_analysis import are_qbits_consecutive, swap_arguments_to_front


def test_are_qbits_consecutive():
    inputs = [1, 0]
    assert are_qbits_consecutive(inputs)

    inputs = [2, 1]
    assert are_qbits_consecutive(inputs)

    inputs = [0, 2]
    assert not are_qbits_consecutive(inputs)


def test_swap_arguments_to_front():
    arguments = [2, 1]
    n = 3
    assert np.array_equal(swap_arguments_to_front(arguments, n)[1], [1, 2, 0])

    arguments = [2, 0]
    assert np.array_equal(swap_arguments_to_front(arguments, n)[1], [0, 2, 1])

    arguments = [1, 2]
    assert np.array_equal(swap_arguments_to_front(arguments, n)[1], [2, 1, 0])

    arguments = [0, 2]
    assert np.array_equal(swap_arguments_to_front(arguments, n)[1], [2, 0, 1])


if __name__ == "__main__":
    test_are_qbits_consecutive()
    test_swap_arguments_to_front()
