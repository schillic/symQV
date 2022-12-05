from quavl.lib.expressions.qbit import Qbits
from quavl.lib.models import Circuit, Method
from quavl.lib.operations import X, H, oracle

if __name__ == "__main__":
    # Initialize circuit
    q0, q1, q2 = Qbits(['q0', 'q1', 'q2'])

    circuit = Circuit([q0, q1, q2],
                      [
                          [H(q0), H(q1), H(q2)],
                          oracle([q0, q1, q2], 5),
                          [H(q0), H(q1), H(q2)],
                          [X(q0), X(q1), X(q2)],
                      ])

    circuit.initialize([(1, 0), (1, 0), (1, 0)])

    # Symbolic execution
    # q0_final, q1_final, q2_final = circuit.get_final_qbits()


    # TODO: oracle also UNSAT for state model
    # Hypothesis 1: Solver can't separate states back into individual qbits,
    #               because works when not getting output qbits.
    # Hypothesis 2: ???

    circuit.prove(method=Method.state_model, dump_smt_encoding=True, dump_solver_output=True)
