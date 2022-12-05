from typing import List, Tuple

import numpy as np

from symqv.lib.constants import I_matrix, SWAP_matrix
from symqv.lib.utils.arithmetic import kron
from symqv.lib.utils.helpers import identity_pad_gate


def normalize(vector: np.array) -> np.array:
    """
    Normalize a complex vector.
    :param vector: complex vector.
    :return: Normalized vector.
    """
    return vector / np.linalg.norm(vector)


def are_qbits_consecutive(inputs: List[int]) -> bool:
    """
    True if the qbit indices are consecutive, false otherwise.
    :param inputs: input qbit indicies.
    :return: Truth value.
    """
    if len(inputs) == 0:
        raise Exception(f'Length {len(inputs)} not supported.')

    if len(inputs) == 1:
        return True

    for i in range(len(inputs) - 1):
        if inputs[i] != inputs[i + 1] + 1:
            return False

    return True


def swap_arguments_to_front(inputs: List[int], num_qbits: int) -> Tuple[np.array, List[int]]:
    """
    Given a list of input qbits (order matters) and the number of qbits in the circuit,
    output a matrix that swaps the input qbits to the front such that they are consecutive and
    a multi-qbit gate can be applied to them.
    :param inputs: input qbits.
    :param num_qbits: number of qbits in circuit.
    :return: Swap matrix and list of new positions.
    """
    swap_matrix = kron([I_matrix] * num_qbits)
    qbits = list(range(num_qbits))

    num_swaps = 0

    for i, input_el in enumerate(inputs):
        for j in range(input_el + num_swaps, 0, -1):
            padded_op = identity_pad_gate(SWAP_matrix, [j - 1, j], num_qbits)
            swap_matrix = np.matmul(padded_op, swap_matrix)

            qbits[j], qbits[j - 1] = qbits[j - 1], qbits[j]

        if i + 1 < len(inputs) and input_el > inputs[i + 1]:
            num_swaps += 1

    return swap_matrix, qbits
