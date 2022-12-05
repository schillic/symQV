from numpy import exp, pi, array, sqrt

import numpy as np

from quavl.lib.utils.arithmetic import kron

q_in = [0, 0, 0, 1]


def to_bin(start, end):
    value = int(''.join([str(b) for b in q_in[start:end]]), 2)

    length = end - start

    print('len', length)
    print('val', value)
    print('result', (1 / (2 ** length)) * value)
    return value


q_out = [1 / sqrt(2) * array([[1], [exp(2 * pi * 1j * (1 / (2 ** 4)) * to_bin(0, 4))]]),
         1 / sqrt(2) * array([[1], [exp(2 * pi * 1j * (1 / (2 ** 3)) * to_bin(1, 4))]]),
         1 / sqrt(2) * array([[1], [exp(2 * pi * 1j * (1 / (2 ** 2)) * to_bin(2, 4))]]),
         1 / sqrt(2) * array([[1], [exp(2 * pi * 1j * (1 / (2 ** 1)) * to_bin(3, 4))]])]

# q_out.reverse()

for v in q_out:
    print(np.array(v).flatten())

print(kron(q_out))
