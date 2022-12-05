import time

# importing Qiskit
from qiskit import QuantumCircuit
from qiskit import quantum_info as qi

qc = QuantumCircuit(3)
qc.ccx(0, 1, 2)


start = time.time()
statevector = qi.Statevector.from_instruction(qc)
# print(statevector)
runtime = time.time() - start

print(f'Runtime raw {runtime}, enumeration {runtime * 2 ** 3}')
