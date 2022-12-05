import numpy as np

from symqv.lib.constants import zero, one
from symqv.lib.operations.state_decomposition import separate_kth_qbit_from_state, from_angles
from symqv.lib.parsing.open_qasm import read_from_qasm
from symqv.lib.utils.arithmetic import kron

teleportation_file = '../../../benchmarks/symqv/teleportv3.qasm'


def test_read_from_qasm():
    qc = read_from_qasm('../../../benchmarks/symqv/cx.qasm')
    initial_state = kron([zero, zero, one])

    final_states = qc.get_final_states(initial_state)
    assert np.allclose(final_states, kron([zero, one, one]))


def test_read_from_qasm_teleportation():
    qc = read_from_qasm(teleportation_file)

    psi = from_angles(np.pi / 4, np.pi / 2)
    initial_state = kron([zero, zero, psi])
    print(initial_state)

    final_states = qc.get_final_states(initial_state)

    for i, final_state in enumerate(final_states):
        print(f'state {i}')
        print(final_state)
        output_qbit, separable, result = separate_kth_qbit_from_state(final_state, 0)

        phi, theta = output_qbit[0], output_qbit[1]

        cartesian = from_angles(phi, theta)

        alpha = cartesian[0, 0]
        beta = cartesian[1, 0]
        print(f"phi = {'{0:.4f}'.format(phi)}, theta = {'{0:.4f}'.format(theta)}")
        print(f"alpha = {'{0:.4f}'.format(alpha.real)} + {'{0:.4f}'.format(alpha.imag)}j")
        print(f"beta = {'{0:.4f}'.format(beta.real)} + {'{0:.4f}'.format(beta.imag)}j")


if __name__ == "__main__":
    test_read_from_qasm_teleportation()
