import collections
import subprocess
import tempfile
import time
from enum import Enum
from sys import platform
from typing import List, Union, Dict, Tuple, Set, Optional

import numpy as np
import pyparsing
from z3 import Solver, Not, And, Or

from symqv.lib.expressions.complex import ComplexVal, Complexes
from symqv.lib.expressions.qbit import QbitVal, Qbits
from symqv.lib.expressions.rqbit import RQbitVal
from symqv.lib.globals import precision_format
from symqv.lib.models.qbit_sequence import QbitSequence
from symqv.lib.models.state_sequence import StateSequence
from symqv.lib.utils.arithmetic import state_not_equals, matrix_vector_multiplication, state_equals, qbit_kron_n_ary, \
    qbit_isclose_to_value
from symqv.lib.utils.helpers import build_qbit_constraints, to_complex_matrix, pi

z3_path = '/usr/local/bin/z3'
dreal_path = None

if platform == "linux" or platform == "linux2":
    dreal_path = '/opt/dreal/4.21.06.2/bin/dreal'
elif platform == "darwin":
    dreal_path = '/usr/local/Cellar/dreal/4.21.06.2/bin/dreal'
elif platform == "win32":
    raise Exception(f'Windows is not directly supported. '
                    f'You need to obtain dReal4 as a Docker container and manually pass .smt2 files to it.')
else:
    raise Exception(f'{platform} not supported. ')


class SpecificationType(Enum):
    transformation_matrix = 'transformation_matrix'
    final_state_vector = 'final_state_vector'
    equality_pair = 'equality_pair'
    equality_pair_list = 'equality_pair_list',
    final_state_qbits = 'final_state_qbits'


def solve(solver: Solver,
          qbits: Union[List[str], List[QbitVal]],
          state_sequence: Union[StateSequence, QbitSequence],
          specification: Union[List, np.ndarray],
          specification_type: SpecificationType,
          is_equality_specification: bool = True,
          output_qbits: List[str] = None,
          delta: float = 0.0001,
          synthesize_repair: bool = False,
          overapproximation: bool = False,
          dump_smt_encoding: bool = False,
          dump_solver_output: bool = False) -> Tuple[str, Union[collections.OrderedDict, None]]:
    """
    Solve for a solver s and a list of variables vars.
    :param solver: solver.
    :param qbits: qbit variables.
    :param state_sequence: state variables.
    :param specification: output specification given as transformation matrix, final state vector or equality pair.
    :param is_equality_specification: whether this is an equality (true) or an inequality (false) specification.
    :param output_qbits: output qbit variables (optional).
    :param delta: error bound.
    :param synthesize_repair: True means the specification is not negated.
    :param overapproximation:
    :param dump_smt_encoding: print the SMT encoding.
    :param dump_solver_output: print the verbatim solver output.
    :return: SAT + counterexample or UNSAT.
    """
    temp_file, qbit_identifiers = write_smt_file(solver,
                                                 qbits,
                                                 state_sequence,
                                                 specification,
                                                 specification_type,
                                                 is_equality_specification,
                                                 output_qbits,
                                                 synthesize_repair,
                                                 delta,
                                                 overapproximation,
                                                 dump_smt_encoding=dump_smt_encoding)

    return run_decision_procedure(temp_file.name,
                                  qbit_identifiers,
                                  state_sequence,
                                  specification,
                                  specification_type,
                                  output_qbits,
                                  delta,
                                  dump_solver_output)


def write_smt_file(solver: Solver,
                   qbits: Union[List[str], List[QbitVal]],
                   state_sequence: Optional[Union[StateSequence, QbitSequence]],
                   specification: Optional[Union[List, np.ndarray]],
                   specification_type: SpecificationType,
                   is_equality_specification: bool = True,
                   output_qbits: List[str] = None,
                   synthesize_repair: bool = False,
                   delta: float = 0.0001,
                   overapproximation: bool = False,
                   dump_smt_encoding: bool = False) -> Tuple[tempfile.NamedTemporaryFile, Set[str]]:
    """
    :param solver: solver.
    :param qbits: qbit variables.
    :param state_sequence: state variables.
    :param specification: output specification given as transformation matrix, final state vector or equality pair.
    :param specification_type: Type of specification.
    :param is_equality_specification: whether this is an equality (true) or an inequality (false) specification.
    :param output_qbits: output qbit variables (optional).
    :param synthesize_repair: True means the specification is not negated.
    :param delta: error bound.
    :param overapproximation: whether to overapproximate qbit angle constraints.
    :param dump_smt_encoding: print the SMT encoding.
    :return: File and qbit identifiers.
    """
    if synthesize_repair:
        is_equality_specification = not is_equality_specification

    # 1 add specification to solver
    if specification is not None:
        if specification_type == SpecificationType.equality_pair \
                or specification_type == SpecificationType.equality_pair_list:
            if specification_type == SpecificationType.equality_pair:
                equality_pair_list = [specification]
            else:
                equality_pair_list = specification

            for pair in equality_pair_list:
                # Equality pair
                (qbit, final_qbit) = pair

                equalities = []

                if isinstance(final_qbit, QbitVal):
                    # one qbit is equal to another one
                    equalities.append(qbit.isclose(final_qbit, delta))
                elif type(final_qbit) == tuple:
                    # one qbit is equal to a specific value
                    equalities.append(qbit_isclose_to_value(qbit, final_qbit, delta))
                else:
                    raise Exception(f'Unsupported specification type {type(final_qbit)}.')

                if is_equality_specification:
                    solver.add(Or([Not(v) for v in equalities]))
                else:
                    solver.add(And(equalities))
        elif specification_type == SpecificationType.final_state_vector:
            # Final state vector is given directly
            if isinstance(state_sequence, StateSequence):
                if is_equality_specification:
                    # Last state may not be equal to specification
                    if isinstance(state_sequence.states[-1], List) and isinstance(state_sequence.states[-1][0], List):
                        # Measurements
                        for state_element in state_sequence.states[-1]:
                            solver.add(state_not_equals(state_element, specification))
                    else:
                        # No Measurements
                        solver.add(
                            state_not_equals(state_sequence.states[-1], specification))
                else:
                    if isinstance(state_sequence.states[-1], List):
                        # Measurements
                        for state_element in state_sequence.states[-1]:
                            solver.add(state_equals(state_element, specification))
                    else:
                        # No Measurements
                        solver.add(state_equals(state_sequence.states[-1], specification))
            else:
                raise Exception('QbitSequence is not supported for specification type final state vector.')
        elif specification_type == SpecificationType.transformation_matrix:
            # Matrix specifies transformation from initial into final state
            if isinstance(state_sequence, StateSequence):
                if is_equality_specification:
                    solver.add(
                        state_not_equals(state_sequence.states[-1],
                                         matrix_vector_multiplication(to_complex_matrix(specification),
                                                                      state_sequence.states[0])))
                else:
                    solver.add(
                        state_equals(state_sequence.states[-1],
                                     matrix_vector_multiplication(to_complex_matrix(specification),
                                                                  state_sequence.states[0])))
            else:
                if not is_equality_specification:
                    raise Exception('QbitSequence not supported for spec type matrix and inequality spec.')

                solver.add(state_not_equals(qbit_kron_n_ary(state_sequence.states[-1]),
                                            matrix_vector_multiplication(to_complex_matrix(specification),
                                                                         qbit_kron_n_ary(state_sequence.states[0]))))
        else:
            raise Exception(f'Specification type {specification_type} is not supported.')

    smt_expr = solver.sexpr()

    smt_expr = smt_expr.replace('(declare-fun sin (Real) Real)\n', '')
    smt_expr = smt_expr.replace('(declare-fun cos (Real) Real)\n', '')
    smt_expr = smt_expr.replace('(declare-fun pi () Real)\n', f'(declare-fun pi () Real)\n(assert (= pi {pi}))')

    smt_expr += '\n'

    # 2 Constrains degrees of freedom
    qbit_identifiers = set()

    # 2.1 build input qbit constraints
    is_reduced_state_space = state_sequence.reduced_state_space

    if isinstance(qbits[0], str):
        qbit_identifiers = set(qbits)
        smt_expr += build_qbit_constraints(qbits, is_reduced_state_space, overapproximation)
    elif isinstance(qbits[0], QbitVal):
        for qbit in qbits:
            qbit_identifiers.add(qbit)
        smt_expr += build_qbit_constraints(list(qbit_identifiers), is_reduced_state_space, overapproximation)

    smt_expr += '\n'

    # 2.2 build output qbit constraints (if used)
    if output_qbits is not None:
        smt_expr += build_qbit_constraints(output_qbits, is_reduced_state_space, overapproximation)

    # 3 Sat strategy
    solver_params = '\n'
    solver_params += ';Solver parameters\n'
    solver_params += ';-----------------\n'
    solver_params += '(check-sat)\n'
    solver_params += '(get-model)\n'
    solver_params += '(exit)\n'

    if dump_smt_encoding:
        print(smt_expr + solver_params)

    # To file
    temp_file = tempfile.NamedTemporaryFile(mode='w+b', suffix='.smt2', delete=False)

    with open(temp_file.name, "w") as text_file:
        text_file.write(smt_expr + solver_params)

    return temp_file, qbit_identifiers


def run_decision_procedure(temp_file_name: str,
                           qbit_identifiers: Set[str],
                           state_sequence: Optional[Union[StateSequence, QbitSequence]],
                           specification: Optional[Union[List, np.ndarray]],
                           specification_type: SpecificationType,
                           output_qbits,
                           delta: float = 0.0001,
                           dump_solver_output=False) -> Tuple[str, Union[collections.OrderedDict, None]]:
    print('Starting solver...')

    sat_result = ''
    model_dict = collections.OrderedDict({})

    # Run
    command = [dreal_path, '--precision', str(delta), temp_file_name] \
        if not isinstance(state_sequence.qbits[0], RQbitVal) else [z3_path, temp_file_name]

    result = subprocess.run(command, stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8')

    # Print output if desired
    if dump_solver_output:
        print(output)

    # Parse output
    output_clean_lines = ''

    for idx, line in enumerate(output.splitlines()):
        if any(line.startswith(s) for s in ['sat', 'delta-sat', 'unsat', 'unknown']):
            sat_result = line.replace('delta', 'δ')
        if "(error" not in line and not line.startswith("(model"):
            output_clean_lines += line + "\n"

    for match in pyparsing.nestedExpr('(', ')').searchString(output_clean_lines):
        key = match[0][1]

        if state_sequence.reduced_state_space:
            value = [bool(str(v).replace('[', '').replace(']', '').replace(',', '')) for v in match[0][4:]]
        else:
            try:
                value = [float(str(v).replace('[', '').replace(']', '').replace(',', '')) for v in match[0][4:]]
            except ValueError:
                value = [True if (str(v).replace('[', '').replace(']', '').replace(',', '')) == 'true' else False
                         for v in match[0][4:]]

        value = value[0] if len(value) == 1 else value

        model_dict[key] = value

    model_dict = dict(sorted(model_dict.items()))

    # Print parsed output
    if state_sequence is not None and isinstance(state_sequence, StateSequence):
        model_dict = _dict_group(model_dict, list(qbit_identifiers), output_qbits, state_sequence)

    if dump_solver_output:
        if len(model_dict) > 0:
            print('Model:')

            for key, value in model_dict.items():
                if isinstance(value, float):
                    print(('{}: ' + precision_format).format(key, value))
                elif isinstance(value, list):
                    lower = value[0]
                    upper = value[1]
                    mean = (lower + upper) / 2
                    error = (upper - lower) / 2

                    print(('{}: ' + precision_format + ' ±{:.16f}').format(key, mean, error))
                else:
                    print('{}: {}'.format(key, repr(value)))

    return sat_result, model_dict


def _dict_group(dict: Dict,
                qbit_identifiers: List[str],
                output_qbits: List[str],
                state_sequence: StateSequence) -> collections.OrderedDict:
    """
    Get an SMT model dict and group by qbits and state.
    :param dict: Model dict.
    :param qbit_identifiers: input qbit identifiers.
    :param output_qbits: output qbit identifiers.
    :param state_sequence: States.
    :return: grouped dict.
    """
    grouped_dict = collections.OrderedDict({})

    if not dict.keys():
        return grouped_dict

    # get states
    for i, state in enumerate(state_sequence.states):
        if isinstance(state[0], List):
            for j, measured_state in enumerate(state):
                for b, state_vector_entry in enumerate(measured_state):
                    key = state_vector_entry.get_identifier().split('.')[0]
                    real = np.mean(dict[f'{key}.r'])
                    imag = np.mean(dict[f'{key}.i'])
                    grouped_dict[f'{key}'] = ComplexVal(real, imag)
        else:
            for j, state_vector_entry in enumerate(state):
                key = state_vector_entry.get_identifier().split('.')[0]
                real = np.mean(dict[f'{key}.r'])
                imag = np.mean(dict[f'{key}.i'])
                grouped_dict[f'{key}'] = ComplexVal(real, imag)

    # get input and output qbits
    all_qbit_keys = qbit_identifiers if output_qbits is None else qbit_identifiers + output_qbits

    for qbit_key in all_qbit_keys:
        alpha_real = np.mean(dict[f'{qbit_key}.alpha.r'])
        alpha_imag = np.mean(dict[f'{qbit_key}.alpha.i'])
        beta_real = np.mean(dict[f'{qbit_key}.beta.r'])
        beta_imag = np.mean(dict[f'{qbit_key}.beta.i'])
        phi = None  # np.mean(dict[f'{qbit_key}.phi'])
        theta = None  # np.mean(dict[f'{qbit_key}.theta'])

        grouped_dict[qbit_key] = QbitVal(ComplexVal(alpha_real, alpha_imag),
                                         ComplexVal(beta_real, beta_imag),
                                         phi, theta)

    return grouped_dict


def reverse_kronecker_product(kron_state: np.ndarray, delta=0.0001) -> List[QbitVal]:
    """
    Reverse a Kronecker product using SMT.
    :param kron_state: numpy Kronecker state vector.
    :param delta: precision.
    :return: Individual qbits.
    """
    solver = Solver()

    kron_state_complexes = Complexes([f'psi{i}' for i in range(kron_state.shape[0])])

    for i in range(len(kron_state)):
        solver.add(kron_state_complexes[i] == kron_state[i, 0])

    n = int(np.log2(kron_state.shape[0]))
    qbit_identifiers = [f'q{i}' for i in range(n)]
    output_qbits = Qbits(qbit_identifiers)

    solver.add(state_equals(kron_state_complexes, qbit_kron_n_ary(output_qbits)))

    smt_expr = solver.sexpr()
    smt_expr += '\n'

    smt_expr += build_qbit_constraints(qbit_identifiers)
    solver_params = '\n'

    solver_params += ';Solver parameters\n'
    solver_params += ';-----------------\n'
    solver_params += '(check-sat)\n'
    solver_params += '(get-model)\n'
    solver_params += '(exit)\n'

    # To file
    temp_file = tempfile.NamedTemporaryFile(mode='w+b', suffix='.smt2', delete=False)

    with open(temp_file.name, "w") as text_file:
        text_file.write(smt_expr + solver_params)

    # Run
    result = subprocess.run([dreal_path, '--precision', str(delta), temp_file.name], stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8')

    if '(error "model is not available")' in output:
        raise Exception('Model is not available.')

    # Parse output
    output_lines = output.split('\n')

    qbit_vals = []

    for q in qbit_identifiers:
        q_lines = [line for line in output_lines if line.strip().startswith(f'(define-fun {q}')]
        qbit_vals.append(_qbit_from_output(q_lines))

    return qbit_vals


def _qbit_from_output(q_lines: List[str]) -> QbitVal:
    """
    Get QbitVal from solver output lines.
    :param q_lines: lines only containing solver output for one qbit.
    :return: QbitVal.
    """

    def to_float(line: str) -> float:
        """
        dReal range to float.
        :param line: dReal model output line.
        :return: float.
        """
        elements = line.strip().split(' ')

        if len(elements) == 5:
            return float(elements[4].replace(')', ''))
        elif len(elements) == 6:
            lower = float(elements[4].replace('[', '').replace(',', ''))
            upper = float(elements[5].replace('])', ''))
            return (lower + upper) / 2
        else:
            print(elements)
            raise Exception('Parser error.')

    alpha_r = to_float([v for v in q_lines if '.alpha.r' in v][0])
    alpha_i = to_float([v for v in q_lines if '.alpha.i' in v][0])
    beta_r = to_float([v for v in q_lines if '.beta.r' in v][0])
    beta_i = to_float([v for v in q_lines if '.beta.i' in v][0])
    phi = to_float([v for v in q_lines if '.phi' in v][0])
    theta = to_float([v for v in q_lines if '.theta' in v][0])

    return QbitVal(alpha=ComplexVal(alpha_r, alpha_i),
                   beta=ComplexVal(beta_r, beta_i),
                   phi=phi,
                   theta=theta)


def _is_sat_result_plausible(model_dict: collections.OrderedDict, specification: Union[List, np.ndarray],
                             num_states: int) -> Tuple[bool, Union[List, None], str]:
    """
    Check if the SAT result returned by the solver is plausible, applying the specification to the initial state
    (in case it is given as a transformation matrix) or by comparing it directly.
    :param model_dict:    values of the model.
    :param specification: specification as either an equality tuple, a transformation matrix or
                          as the final state vector.
    :param num_states:    number of states in the program model.
    :return: true or false.
    """
    initial_state = np.array([[np.complex(model_dict[a].r, model_dict[a].i)
                               for a in model_dict if 'psi_0_' in a]]).T

    final_state = np.array([[np.complex(model_dict[a].r, model_dict[a].i)
                             for a in model_dict if f'psi_{num_states - 1}_' in a]]).T

    if specification is None:
        raise Exception('Specification is not set!')

    if type(specification) == tuple:
        # Final state is given by equality tuple
        qbit = model_dict[specification[0].get_identifier()]

        if type(specification[1]) == tuple:
            value_qbit = specification[1]

            if qbit.alpha.r == value_qbit[0] and qbit.alpha.i == 0 \
                    and qbit.beta.r == value_qbit[1] and qbit.beta.i == 0:
                return True, None, 'Specified qbits is identical to desired value'
        else:
            final_qbit = model_dict[specification[1].get_identifier()]

            if qbit.alpha.r == final_qbit.alpha.r and qbit.alpha.i == final_qbit.alpha.i and \
                    qbit.beta.r == final_qbit.alpha.r and qbit.beta.i == final_qbit.beta.i and \
                    qbit.phi == final_qbit.phi and qbit.theta == final_qbit.theta:
                return True, None, 'Specified input and output qbits are identical'
    elif isinstance(specification, list):
        pass
    elif len(specification.shape) == 1:
        # Final state vector is given directly
        if (final_state == specification).all():
            return True, None, 'Final state is identical to specification'
    else:
        # Matrix specifies transformation from initial into final state
        algebraic_final_state = np.matmul(specification, initial_state)

        if (final_state == algebraic_final_state).all():
            return True, None, 'Final state is result of applying specification to initial state'

    return False, [ComplexVal(np.complex(s).real, np.complex(s).imag) for s in
                   final_state.T[0]], 'Specification is unrelated to final state'
