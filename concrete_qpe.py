# initialization
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import math

# importing Qiskit
from qiskit import IBMQ, Aer, transpile, assemble
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import quantum_info as qi

# import basic plot tools
from qiskit.circuit.library import ZGate
from qiskit.visualization import plot_histogram


def qft_dagger(qc, n):
    """
    Perform n-qubit inverse QFT on the first n qubits in circuit.
    """
    # Don't forget the Swaps!
    for qubit in range(n // 2):
        qc.swap(qubit, n - qubit - 1)

    for j in range(n):
        for m in range(j):
            # controlled phase-gate
            qc.cp(-math.pi / float(2 ** (j - m)), m, j)
        qc.h(j)


def qpe(num_bits=3, theta=0.234):
    # Value of θ which appears in the definition of the unitary U above.
    # Try different values.

    # Define the unitary U.
    u_gate = ZGate().power(2 * theta)

    # Accuracy of the estimate for theta. Try different values.

    # Get qubits for the phase estimation circuit.
    qpe = QuantumCircuit(num_bits + 1, num_bits)
    u_bit = num_bits

    # Build the first part of the phase estimation circuit.
    for i in range(num_bits):
        qpe.h(i)

    qpe.x(u_bit)

    print(u_gate.to_matrix())

    for i, bit in enumerate(range(num_bits)):
        cu_gate = u_gate.control(1).power(2 ** (num_bits - i - 1))
        qpe.append(cu_gate, [bit, u_bit])

    # Do the inverse QFT.
    qpe.barrier()

    # Apply inverse QFT
    qft_dagger(qpe, num_bits)

    print(qpe.draw())

    # Get state vector before measurement
    statevector = qi.Statevector.from_instruction(qpe)
    print(statevector)

    # Measure
    qpe.barrier()
    for i in range(num_bits - 1):
        qpe.measure(i, i)

    aer_sim = Aer.get_backend('aer_simulator')
    shots = 2048
    t_qpe = transpile(qpe, aer_sim)
    qobj = assemble(t_qpe, shots=shots)
    results = aer_sim.run(qobj).result()
    answer = results.get_counts()

    plot_histogram(answer)
    plt.show()

    # Convert from output bitstrings to estimate θ values.
    print(answer)

    estimated_theta = 0

    for key in answer.keys():
        estimated_theta += float(int(key, 2)) * (answer[key] / shots) / 2 ** num_bits

    print(estimated_theta)


if __name__ == '__main__':
    num_shots = 100

    elapsed = 0

    for _ in range(num_shots):
        start = time.time()
        qpe(5, random.uniform(0, np.pi))
        elapsed += time.time() - start

    elapsed = elapsed / num_shots
    print(f'elapsed = {elapsed}')

    delta = 0.0001

    theta_range = np.arange(0, np.pi, delta)

    print(f'num values in range = {len(theta_range)}')

    print(f'time for range = {elapsed * len(theta_range)}')
