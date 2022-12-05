import numpy as np
from qiskit import QuantumCircuit, Aer

from symqv.lib.constants import zero, one, H_matrix, Rk_matrix
from symqv.lib.models.specification import Specification
from symqv.lib.parsing.open_qasm import read_from_qasm
from symqv.lib.utils.arithmetic import kron
from symqv.lib.utils.helpers import identity_pad_gate
from symqv.lib.verification.verifier import verify, build_objective_function


def test_objective_function():
    qc_path = '../../../benchmarks/symqv/teleportv3.qasm'
    spec_path = '../../../benchmarks/symqv/specs/teleportv3.qspec'

    qc = read_from_qasm(qc_path)
    spec = Specification.read_from_file(spec_path)

    objective_function = build_objective_function(qc, spec)

    print(objective_function([0, 0]))
    print(objective_function([2 * np.pi, 0]))
    print(objective_function([0, np.pi]))
    print(objective_function([2 * np.pi, np.pi]))


def test_verify_teleportation():
    qc_path = '../../../benchmarks/symqv/teleportv3.qasm'
    spec_path = '../../../benchmarks/symqv/specs/teleportv3.qspec'

    qc = read_from_qasm(qc_path)
    spec = Specification.read_from_file(spec_path)

    verification_result, counterexample = verify(qc, spec)
    print(f'Verified: {verification_result}, counterexample: {counterexample}')


def test_verify_qft():
    qc_path = '../../../benchmarks/symqv/qft2.qasm'
    spec_path = '../../../benchmarks/symqv/specs/qft.qspec'

    qc = read_from_qasm(qc_path)
    spec = Specification.read_from_file(spec_path)

    verification_result, counterexample = verify(qc, spec)

    input_qbits = [zero, zero, zero, one]
    state = kron(input_qbits)
    print(state)

    print('qc final states')
    print(qc.get_final_states(state))

    # manually apply gates
    state = np.matmul(identity_pad_gate(H_matrix, [0], 4), state)

    if np.allclose(input_qbits[1], one):
        state = np.matmul(identity_pad_gate(Rk_matrix(2), [0], 4), state)
    if np.allclose(input_qbits[2], one):
        state = np.matmul(identity_pad_gate(Rk_matrix(3), [0], 4), state)
    if np.allclose(input_qbits[3], one):
        state = np.matmul(identity_pad_gate(Rk_matrix(4), [0], 4), state)

    state = np.matmul(identity_pad_gate(H_matrix, [1], 4), state)

    if np.allclose(input_qbits[2], one):
        state = np.matmul(identity_pad_gate(Rk_matrix(2), [1], 4), state)
    if np.allclose(input_qbits[3], one):
        state = np.matmul(identity_pad_gate(Rk_matrix(3), [1], 4), state)

    state = np.matmul(identity_pad_gate(H_matrix, [2], 4), state)

    if np.allclose(input_qbits[3], one):
        state = np.matmul(identity_pad_gate(Rk_matrix(2), [2], 4), state)

    state = np.matmul(identity_pad_gate(H_matrix, [3], 4), state)

    print('manually calculated final state')
    print(state)

    if False:
        # check with qiskit
        qc = QuantumCircuit.from_qasm_file(qc_path)

        simulator = Aer.get_backend('statevector_simulator')
        result = simulator.run(qc).result()
        statevector = result.get_statevector(qc)
        print('Qiskit vector')
        print(statevector)


if __name__ == '__main__':
    test_objective_function()
