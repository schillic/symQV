from z3 import Not, Sum, Int

from quavl.lib.constants import to_int, to_bool
from quavl.lib.expressions.qbit import Qbit
from quavl.lib.models.circuit import Method, Circuit
from quavl.lib.operations.gates import V
from quavl.lib.parsing.rev_lib import to_circuit, to_specification


def prove_add64():
    file = 'benchmarks/revlib/add64_186.real'

    # Read circuit from file
    circuit = to_circuit(file)

    initial_values = [(1, 0), {(1, 0), (0, 1)}, {(1, 0), (0, 1)}] * 63 \
                     + [(1, 0), (1, 0), {(1, 0), (0, 1)}, {(1, 0), (0, 1)}]

    circuit.initialize(initial_values)

    qbits = circuit.qbits

    final_qbits = circuit.get_final_qbits()

    circuit.solver.add(to_int(False) == 0)
    circuit.solver.add(to_int(True) == 1)
    circuit.solver.add(to_bool(0) == False)
    circuit.solver.add(to_bool(1) == True)

    circuit.prove(method=Method.qbit_sequence_model, dump_solver_output=False)


def prove_add8():
    file = 'benchmarks/revlib/add8_173.real'

    # Read circuit from file
    circuit = to_circuit(file)

    qbits = circuit.qbits

    [_, y7, x7, _, y6, x6, _, y5, x5, _, y4, x4, _, y3, x3, _, y2, x2, _, y1, x1, _, cin, y0, x0] = qbits

    initial_values = [(1, 0), {(1, 0), (0, 1)}, {(1, 0), (0, 1)}] * 7 \
                     + [(1, 0), (1, 0), {(1, 0), (0, 1)}, {(1, 0), (0, 1)}]

    circuit.initialize(initial_values)

    final_qbits = circuit.get_final_qbits()

    circuit.solver.add(to_int(False) == 0)
    circuit.solver.add(to_int(True) == 1)
    circuit.solver.add(to_bool(0) == False)
    circuit.solver.add(to_bool(1) == True)

    [cout, _, _, s7, _, _, s6, _, _, s5, _, _, s4, _, _, s3, _, _, s2, _, _, s1, s0, _, _] = final_qbits

    x = [x0, x1, x2, x3, x4, x5, x6, x7]
    x_int = Sum([xi.z1 * 2 ** i for i, xi in enumerate(x)])

    y = [y0, y1, y2, y3, y4, y5, y6, y7]
    y_int = Sum([yi.z1 * 2 ** i for i, yi in enumerate(y)])

    s = [s0, s1, s2, s3, s4, s5, s6, s7, cout]
    s_int = Sum([si.z1 * 2 ** i for i, si in enumerate(s)])

    x_int_val = Int('x_int')
    y_int_val = Int('y_int')
    s_int_val = Int('s_int')

    circuit.solver.add(x_int_val == x_int)
    circuit.solver.add(y_int_val == y_int)
    circuit.solver.add(s_int_val == s_int)

    # 2^7 is the highest possible number, 2^8 with carry out.

    # circuit.solver.add(s_int_val != x_int_val + y_int_val)

    circuit.prove(method=Method.state_model, dump_solver_output=True)


def prove_urf():
    file = 'benchmarks/revlib/urf2_152'

    spec_file = 'benchmarks/revlib/specifications/urf2_73.pla'

    # Read circuit from file
    circuit = to_circuit(file)

    qbits = circuit.qbits
    final_qbits = circuit.get_final_qbits()

    spec = to_specification(spec_file, qbits, final_qbits)

    circuit.solver.add(Not(spec))

    circuit.prove(method=Method.qbit_sequence_model, overapproximation=False)


def prove_hwb():
    files = ['benchmarks/revlib/hwb9_122.real', 'benchmarks/revlib/hwb9_123.real']

    spec_file = 'benchmarks/revlib/specifications/hwb9_65.pla'

    for file in files:
        # Read circuit from file
        circuit = to_circuit(file)

        qbits = circuit.qbits
        final_qbits = circuit.get_final_qbits()

        spec = to_specification(spec_file, qbits, final_qbits)

        circuit.solver.add(Not(spec))

        circuit.prove(method=Method.qbit_sequence_model)


def prove_toffoli():
    files = ['benchmarks/revlib/toffoli_1.real', 'benchmarks/revlib/toffoli_2.real']

    spec_file = 'benchmarks/revlib/specifications/toffoli_1.pla'

    for file in files:
        # Read circuit from file
        circuit = to_circuit(file)

        circuit.initialize([{(1, 0), (0, 1)}] * 3)

        qbits = circuit.qbits
        final_qbits = circuit.get_final_qbits()

        spec = to_specification(spec_file, qbits, final_qbits)
        circuit.solver.add(Not(spec))

        circuit.prove(method=Method.qbit_sequence_model, dump_smt_encoding=False, dump_solver_output=True)


def test():
    x = Qbit('x')

    circuit = Circuit([x],
                      [
                          V(x),
                          V(x)
                      ])

    circuit.initialize([(0, 1)])

    circuit.execute(print_computation_steps=True)


if __name__ == "__main__":
    prove_add8()


# URF 199.7900
# HWB 69.0630