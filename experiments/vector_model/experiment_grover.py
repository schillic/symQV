from symqv.lib.expressions.qbit import Qbits
from symqv.lib.models.circuit import Circuit
from symqv.lib.operations.gates import H, CZ, X, CCZ

from symqv.lib.operations.measurements import measure

# Grover's algorithm
def prove_grover():
    # Initialize circuit
    (q0, q1, q2) = Qbits(['q0', 'q1', 'q2'])

    oracle = create_oracle(q0, q1, q2)

    circuit = Circuit([q0, q1, q2],
                      [
                          [H(q0), H(q1), H(q2)],
                          oracle,
                          [H(q0), H(q1), H(q2)],
                          [X(q0), X(q1), X(q2)],
                          CCZ(q0, q1, q2),
                          [X(q0), X(q1), X(q2)],
                          [H(q0), H(q1), H(q2)],
                          measure([q0, q1, q2])
                      ])

    circuit.initialize([(1, 0), (1, 0), (1, 0)])

    circuit.prove()


def create_oracle(q0, q1, q2):
    return [CZ(q0, q2), CZ(q0, q1)]


if __name__ == "__main__":
    prove_grover()
