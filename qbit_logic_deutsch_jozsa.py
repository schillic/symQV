from typing import List

from quavl.lib.expressions.qbit import Qbits, QbitVal
from quavl.lib.models.circuit import Circuit, Method
from quavl.lib.operations.gates import H, X, CNOT, I, Rx, Rz
from quavl.lib.solver import SpecificationType


def prove_deutsch_jozsa(n: int):
    """
    Correctness proof of the Deutsch-Jozsa algorithm.
    :param n: number of qbits.
    """
    qbits = Qbits([f'q{i}' for i in range(n)])

    for balanced in [True, False]:
        # Create oracles
        oracles = create_oracles(qbits, balanced)
        oracle = oracles[0]

        # Initialize circuit
        circuit = Circuit(qbits,
                          [
                              [H(q) for q in qbits],
                              *oracle,
                              [H(q) for q in qbits],
                              [X(q) for q in qbits[:-1]],
                              X(qbits[-1]).controlled_by(qbits[:-1]),
                          ])

        print('Balanced:' if balanced else 'Constant:')
        print(circuit)

        circuit.initialize([(1, 0) for _ in qbits[:-1]] + [(0, 1)])

        # Symbolic execution
        final_qbits = circuit.get_final_qbits()

        if balanced:
            # balanced oracle: control qbits are not all (0, 1), so last qbit output is unchanged from (0, 1)
            circuit.set_specification((final_qbits[-1], (0, 1)), SpecificationType.equality_pair)
        else:
            # constant oracle: control qbits are all (0, 1), so last qbit output is changed to (1, 0)
            circuit.set_specification((final_qbits[-1], (1, 0)), SpecificationType.equality_pair)

        circuit.prove(method=Method.qbit_sequence_model,
                      dump_smt_encoding=False,
                      overapproximation=True)


def create_oracles(qbits: List[QbitVal], balanced: bool):
    """
    Create all possible balanced or constant oracles.
    :param qbits: qbits in the system.
    :param balanced: true if balanced, false if constant.
    :return: list of oracles.
    """
    if balanced:
        # balanced
        operations = []
        num_digits = len('{0:b}'.format(2 ** len(qbits))) - 1
        binary_format = '{0:0' + str(num_digits) + 'b}'

        for i in range(1, len(qbits) - 1):
            balanced_op = []

            bit_vector = binary_format.format(i)

            for j, b in enumerate(bit_vector[::-1]):
                if b == '1':
                    balanced_op.append(CNOT(qbits[j], qbits[-1]))

            operations.append(balanced_op)
            operations.append(balanced_op + [X(qbits[-1])])

        return operations

    # constant
    return [[I(qbits[-1])], [X(qbits[-1])]]


if __name__ == "__main__":
   prove_deutsch_jozsa(3)


#   n  time [s]: b     c        n_vars   n_f
#   3       3.3387 +   3.1446    168      433
#   4      10.5307 +  10.2110    224      575
#   5      23.0817 +  32.6885    280      717
#   6      69.7740 +  75.2547    336      859
#   7     129.5028 + 133.5520    392     1001
#   8    1160.7479 + 237.2836    448     1143
#   9    5071.4266 + 332.5097    504     1285
#   10  21915.5369 + 479.6665    560     1427
# * 11  48421.6892 + 1587.7856   616     1569
# * 12 168569.4512 + 3209.9019   672     1711


# Over-approximation
#    n   time [s]       n_vars  n_f
#    3   1.4801         168     454
#    4   3.0058         224     603
#    5   5.5697         280     752
#    6   9.0593         336     901
#    7   14.3199        392     1050
#    8   23.0722        448     1199
#    9   38.2228        504     1348
#   10   66.5038        560     1497
#   11  125.3783        616     1646
#   12  251.9041        672     1795
#   13  601.8831        728     1944
#   14 1295.3248        784     2093
#   15 3031.5521        840     2242
#   16 7557.2224        896     2391
#   17 19109.2565       952     2540
#   18 46300.1143       1008    2689
