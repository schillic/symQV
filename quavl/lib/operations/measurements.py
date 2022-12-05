import math
from typing import List, Union

from quavl.lib.expressions.qbit import QbitVal
from quavl.lib.models.measurement import Measurement
import numpy as np


def measure(qbits: Union[QbitVal, List[QbitVal]]) -> Measurement:
    return Measurement(qbits)


def get_measurement_probability_from_state(qbit_index: int, state: np.array) -> float:
    """
    Get the |0> measurement probability from a state and a qbit index.
    :param qbit_index: index of the qbit within the system.
    :param state: state vector.
    :return: probability that given qbit measures 0 in given state vector.
    """
    num_qbits = math.log(len(state), 2)

    if not num_qbits.is_integer():
        raise Exception("Illegal state: Dimension must be a power of 2.")

    binary_indices = []

    for i in range(len(state)):
        binary_indices.append(bin(i)[2:].zfill(int(num_qbits)))

    probability = 0.0

    for i, binary_index in enumerate(binary_indices):
        if binary_index[qbit_index] == '0':
            probability += abs(state[i][0]) ** 2

    return probability


zero_measurement = np.array([[1, 0],
                             [0, 0]])

one_measurement = np.array([[0, 0],
                            [0, 1]])
