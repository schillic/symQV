import time

from numpy import pi
# importing Qiskit
from qiskit import QuantumCircuit
from qiskit import quantum_info as qi


def qft_rotations(circuit, n):
    if n == 0:  # Exit function if circuit is empty
        return circuit
    n -= 1  # Indexes start from 0
    circuit.h(n)  # Apply the H-gate to the most significant qubit
    for qubit in range(n):
        # For each less significant qubit, we need to do a
        # smaller-angled controlled rotation:
        circuit.cp(pi / 2 ** (n - qubit), qubit, n)


def swap_registers(circuit, n):
    for qubit in range(n // 2):
        circuit.swap(qubit, n - qubit - 1)
    return circuit


def qft(circuit, n):
    """QFT on the first n qubits in circuit"""
    qft_rotations(circuit, n)
    swap_registers(circuit, n)
    return circuit


# Let's see how it looks:
n = 12
qc = QuantumCircuit(n)
qft(qc, n)

start = time.time()
statevector = qi.Statevector.from_instruction(qc)
# print(statevector)
runtime = time.time() - start

print(f'Runtime raw {runtime}, enumeration {runtime * 2 ** n}')
