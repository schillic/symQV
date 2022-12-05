from qiskit import QuantumCircuit
from qiskit.ignis.verification import get_ghz_simple

from quavl.lib.expressions.qbit import Qbits
from quavl.lib.models.circuit import Circuit
from quavl.lib.operations.gates import CNOT, H

import qiskit.quantum_info as qi


def execute_ghz(n: int = 3):
    qbits = Qbits([f'q{i}' for i in range(n)])

    program = [H(qbits[0])]

    for i in range(n - 1):
        program.append(CNOT(qbits[i], qbits[i + 1]))

    circuit = Circuit(qbits, program)

    # Concrete execution
    circuit.initialize([(0, 1), (0, 1), (0, 1)])
    output = circuit.execute()

    print(output)


def qiskit_ghz(n: int = 3):
    circ_simple = QuantumCircuit(n)
    circ_simple.h(0)

    for i in range(n - 1):
        circ_simple.cnot(i, i + 1)

    output = qi.Statevector.from_instruction(circ_simple)
    print(output)


if __name__ == '__main__':
    qiskit_ghz(28)
