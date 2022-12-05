import time

import numpy as np
from z3 import Sum, Int

from symqv.lib.constants import to_int, to_bool
from symqv.lib.models.circuit import Method
from symqv.lib.parsing.rev_lib import to_circuit


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

    circuit.prove(method=Method.state_model, dump_solver_output=True)


if __name__ == "__main__":
    times = []

    for _ in range(5):
        start = time.time()
        prove_add8()
        times.append(time.time() - start)

    print(f'Runtime:', np.mean(times))
