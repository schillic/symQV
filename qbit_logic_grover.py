from quavl.lib.expressions.qbit import Qbits
from quavl.lib.models.circuit import Circuit, Method
from quavl.lib.operations.gates import X, H, CCZ, CZ

if __name__ == "__main__":
    # Initialize circuit
    (q0, q1, q2) = Qbits(['q0', 'q1', 'q2'])

    circuit = Circuit([q0, q1, q2],
                      [
                          [H(q0), H(q1), H(q2)],
                          CZ(q0, q2), CZ(q0, q1),
                          [H(q0), H(q1), H(q2)],
                          [X(q0), X(q1), X(q2)],
                          CCZ(q0, q1, q2),
                          [X(q0), X(q1), X(q2)],
                          [H(q0), H(q1), H(q2)]
                      ])

    circuit.initialize([(1, 0), (1, 0), (1, 0)])

    # Symbolic execution
    (q0_final, _, _) = circuit.get_final_qbits()

    circuit.prove(method=Method.qbit_sequence_model,
                  dump_smt_encoding=True,
                  dump_solver_output=True,
                  overapproximation=True)
