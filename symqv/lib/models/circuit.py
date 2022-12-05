import collections
import time
from enum import Enum
from tempfile import NamedTemporaryFile
from typing import List, Tuple, Union, Set

import numpy as np
from z3 import Tactic, Or, And, If, Real, Bool

from symqv.lib.constants import I_matrix, CNOT_matrix, SWAP_matrix
from symqv.lib.expressions.qbit import QbitVal, Qbits
from symqv.lib.expressions.rqbit import RQbitVal, RQbits
from symqv.lib.globals import precision_format
from symqv.lib.models.gate import Gate
from symqv.lib.models.measurement import Measurement
from symqv.lib.models.qbit_sequence import QbitSequence
from symqv.lib.models.state_sequence import StateSequence
from symqv.lib.operations.measurements import zero_measurement, one_measurement, get_measurement_probability_from_state
from symqv.lib.solver import solve, write_smt_file, SpecificationType
from symqv.lib.utils.arithmetic import state_equals, qbit_equals_value, matrix_vector_multiplication, \
    complex_kron_n_ary, kron, state_equals_phase_oracle, rqbit_equals_rqbit
from symqv.lib.utils.helpers import get_qbit_indices, identity_pad_gate, to_complex_matrix, \
    identity_pad_single_qbit_gates, are_qbits_reversed, are_qbits_adjacent, swap_transform_non_adjacent_gate


class Method(Enum):
    state_model = 'state_model'
    qbit_sequence_model = 'qbit_sequence_model'


class Circuit:
    def __init__(self, qbits: List[Union[QbitVal, RQbitVal]],
                 program: List[Union[Gate, List[Gate], Measurement]],
                 delta: float = 0.0001):
        """
        Constructor for a circuit with qbits.
        :param qbits: list of qbits.
        :param: program: list of gates and measurements.
        :param: delta: error bound.
        """
        self.qbits = qbits
        self.final_qbits = None
        self.program = program
        self.delta = delta
        self.initial_qbit_values = None
        self.initial_gate_applications = None
        self.initial_state_value = None
        self.specification = None
        self.specification_type = None
        self.is_equality_specification = None
        self.initialization_has_none_values = False
        self.solver = Tactic('qfnra-nlsat').solver()

    def __str__(self):
        return ', '.join([str(gate) for gate in self.program])

    def initialize(self, values: List[Union[Tuple[Union[int, complex], Union[int, complex]],
                                            Set[Tuple[int, int]]]]):
        """
        Initialize a quantum circuit with qbit values.
        :param values: list of value pairs.
        :return: void.
        """
        if self.initial_qbit_values is not None:
            print('Qbits are already initialized. Reinitializing.')

        for i, value in enumerate([v for v in values if v is not None]):
            if type(value) is tuple:
                if not isinstance(self.qbits[0], RQbitVal):
                    (alpha, beta) = value
                    magnitude = abs(alpha) ** 2 + abs(beta) ** 2

                    if magnitude < 1.0 - self.delta or 1.0 + self.delta < magnitude:
                        raise Exception(f'Illegal qbit magnitude: was {magnitude}, must be 1.')
                else:
                    if value != (1, 0) and value != (0, 1):
                        raise Exception(f'Illegal qbit value: was {value}, must be (1, 0) or (0, 1).')

                self.solver.add(qbit_equals_value(self.qbits[i], value))
            elif type(value) is set:
                self.solver.add(Or([qbit_equals_value(self.qbits[i], element) for element in value]))

        self.initialization_has_none_values = any([value is None for value in values])
        self.initial_qbit_values = values

    def set_initial_gate_applications(self, gates: List[Gate]):
        """
        Use gates to construct the initial state.
        :param gates:
        :return:
        """
        if self.initial_qbit_values is None:
            raise Exception("No initial values provided.")

        self.initial_gate_applications = gates

    def set_specification(self,
                          specification,
                          specification_type: SpecificationType,
                          is_equality_specification: bool = True):
        """
        Set a specification.
        :param specification: specification
        :param specification_type: specification's type
        :param is_equality_specification: whether this is an equality (true) or an inequality (false) specification.
        """
        self.specification = specification
        self.specification_type = specification_type
        self.is_equality_specification = is_equality_specification

    def get_final_qbits(self):
        """
        Create only if used.
        :return: final qbits.
        """
        if isinstance(self.qbits[0], RQbitVal):
            self.final_qbits = RQbits([qbit.get_identifier() + '_final' for qbit in self.qbits])

            for final_qbit in self.final_qbits:
                self.solver.add(And(final_qbit.v0 == False, final_qbit.v1 == False))
        else:
            self.final_qbits = Qbits([qbit.get_identifier() + '_final' for qbit in self.qbits])
        return self.final_qbits

    def execute(self, print_computation_steps: bool = False):
        """
        Calculate a quantum circuit's output.
        :return: Calculated output.
        """
        if print_computation_steps:
            print('Concrete execution...\n')

        if self.initial_qbit_values is None and self.initial_state_value is None:
            raise Exception("No initial values provided.")

        if self.initialization_has_none_values:
            raise Exception("Initial values only provided for some qbits.")

        if print_computation_steps:
            print(f'Initial values: {self.initial_qbit_values}')

        # 1 Build initial state
        if self.initial_state_value is None:
            self.initial_state_value = kron([np.array([qbit]).T for qbit in self.initial_qbit_values])

        state = self.initial_state_value

        # 2 Initial gate applications
        if self.initial_gate_applications is not None:
            combined_initial_gate = identity_pad_gate(I_matrix, [0], len(self.qbits))

            for i, gate in enumerate(self.initial_gate_applications):
                # build state operation
                qbit_indices = get_qbit_indices(self.qbits, gate.arguments)

                if are_qbits_reversed(qbit_indices):
                    combined_initial_gate = np.matmul(
                        identity_pad_gate(gate.matrix_swapped, qbit_indices, len(self.qbits)),
                        combined_initial_gate)
                else:
                    combined_initial_gate = np.matmul(
                        identity_pad_gate(gate.matrix, qbit_indices, len(self.qbits)),
                        combined_initial_gate)

            state = np.matmul(combined_initial_gate, state)

        if print_computation_steps:
            print(f'Initial state ψ ≐ {state.T}.T')

        # 3 Compute subsequent states
        for i, operation in enumerate(self.program):
            if isinstance(operation, Gate) or isinstance(operation, List):
                gates = [operation] if isinstance(operation, Gate) else operation

                for gate in gates:
                    if gate.control_qbits is not None:
                        raise Exception('Control qbits not supported')

                    # build state operation
                    qbit_indices = get_qbit_indices(self.qbits, gate.arguments)

                    if gate.oracle_value is None:
                        if are_qbits_reversed(qbit_indices):
                            state_operation = identity_pad_gate(gate.matrix_swapped, qbit_indices,
                                                                len(self.qbits))
                        else:
                            state_operation = identity_pad_gate(gate.matrix, qbit_indices, len(self.qbits))
                        state = np.matmul(state_operation, state)
                    else:
                        raise Exception('Oracle not supported')

            elif isinstance(operation, Measurement):
                # build state operation
                arguments: List[QbitVal] = operation.arguments if isinstance(operation.arguments, List) \
                    else [operation.arguments]

                qbit_indices = get_qbit_indices(self.qbits, arguments)
                probabilities = []

                for qbit_index in qbit_indices:
                    probabilities.append(get_measurement_probability_from_state(qbit_index, state))

                measurement_matrices = []

                for j in range(len(arguments)):
                    zero_probability = probabilities[j]
                    one_probability = 1.0 - zero_probability

                    if print_computation_steps:
                        print(f'Measuring qbit |{arguments[j].get_identifier()}⟩:'
                              f' p(0) = {precision_format.format(zero_probability)},'
                              f' p(1) = {precision_format.format(one_probability)}')

                    measurement = np.random.choice([0, 1], p=[zero_probability, one_probability], size=1)[0]
                    measurement_matrices.append(zero_measurement if measurement == 0 else one_measurement)

                state_operation = identity_pad_single_qbit_gates(measurement_matrices, qbit_indices, len(self.qbits))
                state = np.matmul(state_operation, state)
            else:
                raise Exception(f'Unsupported operation {type(operation)}. Has to be either gate or measurement.')

            if print_computation_steps:
                print(f'ψ_{i} ≐ {state.T}.T')

        return state

    def prove(self,
              dump_smt_encoding: bool = False,
              dump_solver_output: bool = False,
              measurement_branch: int = None,
              file_generation_only: bool = False,
              no_intermediate_state_constraints: bool = False,
              method: Method = Method.state_model,
              synthesize_repair: bool = False,
              repair_parameter_bound: float = 0.1,
              entangling_repair: bool = False,
              entangling_gate_index: int = 0,
              overapproximation: bool = False) -> Union[Tuple[str, collections.OrderedDict, float],
                                                        Tuple[NamedTemporaryFile, Set[str]]]:

        if method == Method.state_model:
            return self._prove_state_model(dump_smt_encoding,
                                           dump_solver_output,
                                           measurement_branch,
                                           file_generation_only,
                                           synthesize_repair,
                                           overapproximation)
        elif method == Method.qbit_sequence_model:
            return self._prove_qbit_sequence_model(dump_smt_encoding,
                                                   dump_solver_output,
                                                   measurement_branch,
                                                   file_generation_only,
                                                   no_intermediate_state_constraints,
                                                   synthesize_repair,
                                                   repair_parameter_bound,
                                                   entangling_repair,
                                                   entangling_gate_index,
                                                   overapproximation)

        else:
            raise Exception(f'Unsupported method {method}.')

    def _prove_qbit_sequence_model(self,
                                   dump_smt_encoding: bool = False,
                                   dump_solver_output: bool = False,
                                   measurement_branch: int = None,
                                   file_generation_only: bool = False,
                                   no_intermediate_state_constraints: bool = False,
                                   synthesize_repair: bool = False,
                                   repair_parameter_bound: float = 0.1,
                                   entangling_repair: bool = False,
                                   entangling_gates: int = 0,
                                   overapproximation: bool = False) \
            -> Union[Tuple[str, collections.OrderedDict, float],
                     Tuple[NamedTemporaryFile, Set[str]]]:
        """
        Prove a quantum circuit according to the qbit sequence model, symbolically encoding states qbit lists.
        :param dump_smt_encoding:  print the utils encoding.
        :param dump_solver_output: print the verbatim solver output.
        :param file_generation_only: only generate file, don't call solver.
        :param synthesize_repair: Synthesize repair to make the circuit fulfill the specification.
        :param entangling_repair: In case of repair synthesis, potentially entangle all qubits.
        :return: Solver output.
        """
        start_full = time.time()

        # 1 Build initial state definition
        qbit_sequence = QbitSequence(self.qbits)

        computational_basis_only = False  # all(g.name == 'X' for g in self.program)

        if isinstance(self.qbits[0], RQbitVal):
            for qbit in self.qbits:
                self.solver.add(qbit.get_constraints(computational_basis_only))

        # 2 Build initial state from gates (optional)
        if self.initial_gate_applications is not None:
            for (i, gate) in enumerate(self.initial_gate_applications):
                previous_state = qbit_sequence.states[-1]
                next_state = qbit_sequence.add_state()

                qbit_indices = get_qbit_indices([q.get_identifier() for q in self.qbits], gate.arguments)
                other_qbit_indices = [i for i, _ in enumerate(self.qbits) if i not in qbit_indices]

                if gate.arity() == 1:
                    self.solver.add(next_state[qbit_indices[0]]
                                    == matrix_vector_multiplication(gate.matrix,
                                                                    previous_state[qbit_indices[0]].to_complex_list()))

                    for i in other_qbit_indices:
                        self.solver.add(next_state[i] == previous_state[i])
                else:
                    raise Exception('Not supported yet')

        # 3 Iterate all operations and build intermediate states
        for (i, gate) in enumerate(self.program):
            # current state vector
            previous_state = qbit_sequence.states[-1]
            next_state = qbit_sequence.add_state()

            if isinstance(self.qbits[0], RQbitVal):
                for qbit in next_state:
                    self.solver.add(qbit.get_constraints(computational_basis_only))

                # single gate
                qbit_indices = get_qbit_indices([q.get_identifier() for q in self.qbits], gate.arguments)
                other_qbit_indices = [i for i, _ in enumerate(self.qbits) if i not in qbit_indices]

                if gate.arity() != 1:
                    raise Exception(f'Arity {gate.arity()} not supported.')

                transformed_qbit_val = gate.r_mapping(previous_state[qbit_indices[0]])

                if gate.control_qbits is None:
                    # no control qbits
                    self.solver.add(next_state[qbit_indices[0]] == transformed_qbit_val)
                else:
                    # gate with control qbits
                    control_qbit_indices = get_qbit_indices([q.get_identifier() for q in self.qbits],
                                                            gate.control_qbits)

                    # all 1 => gate applied
                    # otherwise => identity
                    self.solver.add(
                        If(And([previous_state[j].z1 for j in control_qbit_indices]),
                           rqbit_equals_rqbit(next_state[qbit_indices[0]], transformed_qbit_val),
                           rqbit_equals_rqbit(next_state[qbit_indices[0]], previous_state[qbit_indices[0]])))

                # unchanged qbits
                for j in other_qbit_indices:
                    self.solver.add(next_state[j] == previous_state[j])

                continue

            if isinstance(gate, Gate):
                # single gate
                qbit_indices = get_qbit_indices([q.get_identifier() for q in self.qbits], gate.arguments)
                other_qbit_indices = [i for i, _ in enumerate(self.qbits) if i not in qbit_indices]

                if gate.arity() == 1:
                    # one qbit gate
                    if gate.mapping is None:
                        transformed_qbit = matrix_vector_multiplication(to_complex_matrix(gate.matrix),
                                                                        previous_state[
                                                                            qbit_indices[0]].to_complex_list())

                        transformed_qbit_val = QbitVal(transformed_qbit[0], transformed_qbit[1])
                    else:
                        transformed_qbit_val = gate.mapping(previous_state[qbit_indices[0]])

                    if gate.control_qbits is None:
                        # no control qbits
                        self.solver.add(next_state[qbit_indices[0]] ==
                                        transformed_qbit_val)
                    else:
                        # gate with control qbits
                        control_qbit_indices = get_qbit_indices([q.get_identifier() for q in self.qbits],
                                                                gate.control_qbits)

                        # all 1 => gate applied
                        # 0 and 1 => identity
                        self.solver.add(And(
                            [Or(qbit_equals_value(previous_state[j], (0, 1)),
                                qbit_equals_value(previous_state[j], (1, 0))) for j in control_qbit_indices]))

                        self.solver.add(
                            If(And([previous_state[j].beta.r == 1 for j in control_qbit_indices]),
                               next_state[qbit_indices[0]] == transformed_qbit_val,
                               next_state[qbit_indices[0]] == previous_state[qbit_indices[0]]))

                        # unchanged control qbits already in other_qbit_indices
                else:
                    # multi qbit gate
                    previous_kron_state = complex_kron_n_ary(
                        [previous_state[i].to_complex_list() for i in qbit_indices])
                    next_kron_state = complex_kron_n_ary([next_state[i].to_complex_list() for i in qbit_indices])

                    if gate.oracle_value is None:
                        # (symbolic kronecker product from gate input qbits)
                        self.solver.add(state_equals(next_kron_state,
                                                     matrix_vector_multiplication(to_complex_matrix(gate.matrix),
                                                                                  previous_kron_state)))
                    else:
                        # phase oracle application
                        self.solver.add(
                            state_equals_phase_oracle(next_kron_state, previous_kron_state, gate.oracle_value))

                # unchanged qbits
                for j in other_qbit_indices:
                    self.solver.add(next_state[j] == previous_state[j])
            elif isinstance(gate, List):
                # list of gates
                combined_qbit_indices = []

                for single_gate in gate:
                    combined_qbit_indices.extend(
                        get_qbit_indices([q.get_identifier() for q in self.qbits], single_gate.arguments))

                if len(set(combined_qbit_indices)) < len(combined_qbit_indices):
                    raise Exception('Only pairwise disjoint gates in gate list allowed.')

                other_qbit_indices = [i for i, _ in enumerate(self.qbits) if i not in combined_qbit_indices]

                # transform qbit per gate
                for single_gate in gate:
                    if single_gate.arity() == 1:
                        if single_gate.control_qbits is None:
                            qbit_indices = get_qbit_indices([q.get_identifier() for q in self.qbits],
                                                            single_gate.arguments)

                            if single_gate.mapping is None:
                                transformed_qbit = matrix_vector_multiplication(to_complex_matrix(single_gate.matrix),
                                                                                previous_state[
                                                                                    qbit_indices[0]].to_complex_list())

                                transformed_qbit_val = QbitVal(transformed_qbit[0], transformed_qbit[1])
                            else:
                                transformed_qbit_val = single_gate.mapping(previous_state[qbit_indices[0]])

                            self.solver.add(next_state[qbit_indices[0]] == transformed_qbit_val)
                        else:
                            raise Exception('No controlled qbit gates in List')
                    else:
                        raise Exception('No multi qbit gates in List')

                for j in other_qbit_indices:
                    self.solver.add(next_state[j] == previous_state[j])
            elif isinstance(gate, Measurement):
                qbit_indices = get_qbit_indices([q.get_identifier() for q in self.qbits], gate.arguments)
                other_qbit_indices = [i for i, _ in enumerate(self.qbits) if i not in qbit_indices]

                for j in qbit_indices:
                    self.solver.add(
                        If(previous_state[j].alpha.r == 1,
                           qbit_equals_value(next_state[j], (1, 0)),
                           If(previous_state[j].beta.r == 1,
                              qbit_equals_value(next_state[j], (0, 1)),
                              Or(qbit_equals_value(next_state[j], (1, 0)),
                                 qbit_equals_value(next_state[j], (0, 1))))))

                # unchanged qbits
                for j in other_qbit_indices:
                    self.solver.add(next_state[j] == previous_state[j])
            else:
                raise Exception('Operation type not supported.')

        # 4.1 Repair synthesis:
        if self.final_qbits is not None and synthesize_repair is True:
            print('Repair')

            previous_state = qbit_sequence.states[-1]
            next_state = qbit_sequence.add_state('repair')

            bound = repair_parameter_bound

            # parameters theta, phi, lambda in [-2pi, 2pi] with global phase and [-pi, pi] without
            for i in range(len(self.qbits)):
                theta = Real(f'rep_theta_{i}')
                phi = Real(f'rep_phi_{i}')

                self.solver.add(-bound <= theta)
                self.solver.add(theta <= bound)
                self.solver.add(-bound <= phi)
                self.solver.add(phi <= bound)

                self.solver.add(next_state[i].theta == previous_state[i].theta + theta)
                self.solver.add(next_state[i].phi == previous_state[i].phi + phi)

            if entangling_repair is True:
                print('Entangling')

                if len(self.qbits) != 3:
                    raise Exception(f'Qbit count of {len(self.qbits)} is not supported.')

                if entangling_gates == 0:
                    b1 = Bool('b1')

                    previous_state = qbit_sequence.states[-1]
                    next_state = qbit_sequence.add_state('repair_ent1')

                    previous_kron_state = complex_kron_n_ary(
                        [previous_state[i].to_complex_list() for i in range(len(self.qbits))])
                    next_kron_state = complex_kron_n_ary(
                        [next_state[i].to_complex_list() for i in range(len(self.qbits))])

                    entangling_matrix = identity_pad_gate(CNOT_matrix, [0, 1], len(self.qbits))
                    entangling_matrix = np.matmul(identity_pad_gate(SWAP_matrix, [1, 2], len(self.qbits)),
                                                  entangling_matrix)

                    self.solver.add(If(b1,
                                       state_equals(next_kron_state,
                                                    matrix_vector_multiplication(to_complex_matrix(entangling_matrix),
                                                                                 previous_kron_state)),
                                       state_equals(next_kron_state,
                                                    previous_kron_state)))

                if entangling_gates == 1:
                    b2 = Bool('b2')

                    previous_state = qbit_sequence.states[-1]
                    next_state = qbit_sequence.add_state('repair_ent2')

                    previous_kron_state = complex_kron_n_ary(
                        [previous_state[i].to_complex_list() for i in range(len(self.qbits))])
                    next_kron_state = complex_kron_n_ary(
                        [next_state[i].to_complex_list() for i in range(len(self.qbits))])

                    entangling_matrix = identity_pad_gate(CNOT_matrix, [0, 1], len(self.qbits))
                    entangling_matrix = np.matmul(identity_pad_gate(SWAP_matrix, [1, 2], len(self.qbits)),
                                                  entangling_matrix)

                    self.solver.add(If(b2,
                                       state_equals(next_kron_state,
                                                    matrix_vector_multiplication(to_complex_matrix(entangling_matrix),
                                                                                 previous_kron_state)),
                                       state_equals(next_kron_state,
                                                    previous_kron_state)))

                if entangling_gates == 2:
                    b3 = Bool('b3')

                    previous_state = qbit_sequence.states[-1]
                    next_state = qbit_sequence.add_state('repair_ent3')

                    previous_kron_state = complex_kron_n_ary(
                        [previous_state[i].to_complex_list() for i in range(len(self.qbits))])
                    next_kron_state = complex_kron_n_ary(
                        [next_state[i].to_complex_list() for i in range(len(self.qbits))])

                    entangling_matrix = identity_pad_gate(CNOT_matrix, [1, 2], len(self.qbits))

                    self.solver.add(If(b3,
                                       state_equals(next_kron_state,
                                                    matrix_vector_multiplication(to_complex_matrix(entangling_matrix),
                                                                                 previous_kron_state)),
                                       state_equals(next_kron_state,
                                                    previous_kron_state)))

                previous_state = qbit_sequence.states[-1]
                next_state = qbit_sequence.add_state('repair_2')

                # parameters theta, phi, lambda in [-2pi, 2pi] with global phase and [-pi, pi] without
                for i in range(len(self.qbits)):
                    theta = Real(f'rep2_theta_{i}')
                    phi = Real(f'rep2_phi_{i}')

                    self.solver.add(-bound <= theta)
                    self.solver.add(theta <= bound)
                    self.solver.add(-bound <= phi)
                    self.solver.add(phi <= bound)

                    self.solver.add(next_state[i].theta == previous_state[i].theta + theta)
                    self.solver.add(next_state[i].phi == previous_state[i].phi + phi)

        # 4.2 Final state qbits
        if self.final_qbits is not None:
            final_state = qbit_sequence.states[-1]

            for i in range(len(self.qbits)):
                if isinstance(self.qbits[0], RQbitVal):
                    self.solver.add(self.final_qbits[i].get_constraints(computational_basis_only))
                self.solver.add(final_state[i] == self.final_qbits[i])

        # 5 Call solver
        qbit_identifiers = [qbit.get_identifier() for qbit in self.qbits]

        for state in qbit_sequence.states:
            # add intermediate qbits
            qbit_identifiers.extend([qbit.get_identifier() for qbit in state])

        (sat_result, model) = solve(self.solver,
                                    qbit_identifiers,
                                    qbit_sequence,
                                    self.specification,
                                    self.specification_type,
                                    self.is_equality_specification,
                                    output_qbits=[q.get_identifier() for q in self.final_qbits]
                                    if self.final_qbits is not None else None,
                                    delta=self.delta,
                                    synthesize_repair=self.final_qbits is not None and synthesize_repair is True,
                                    overapproximation=overapproximation,
                                    dump_smt_encoding=dump_smt_encoding,
                                    dump_solver_output=dump_solver_output)

        end_full = time.time()
        time_full = end_full - start_full

        print(f'Elapsed time {precision_format.format(time_full)} seconds.')
        return sat_result, model, time_full

    def _prove_state_model(self,
                           dump_smt_encoding: bool = False,
                           dump_solver_output: bool = False,
                           measurement_branch: int = None,
                           file_generation_only: bool = False,
                           synthesize_repair: bool = False,
                           overapproximation: bool = False) -> Union[Tuple[str, collections.OrderedDict, float],
                                                                     Tuple[NamedTemporaryFile, Set[str]]]:
        """
        Prove a quantum circuit according to the state model, symbolically encoding states as full vectors.
        :param dump_smt_encoding:  print the utils encoding.
        :param dump_solver_output: print the verbatim solver output.
        :param measurement_branch: which measurement branch to consider (optional, only used by parallel evaluation).
        :param file_generation_only: only generate file, don't call solver.
        :param synthesize_repair: Synthesize repair to make the circuit fulfill the specification.
        :return: Solver output.
        """
        start_full = time.time()

        # 1 Build initial state definition
        # (Symbolic kronecker product from input qbits)
        initial_state_definition = complex_kron_n_ary([qbit.to_complex_list() for qbit in self.qbits])

        state_sequence = StateSequence(self.qbits)

        # 2 Build initial state from gates (optional)
        if self.initial_gate_applications is not None:
            combined_initial_gate = identity_pad_gate(I_matrix, [0], len(self.qbits))

            for (i, gate) in enumerate(self.initial_gate_applications):
                # build combined initial gate
                qbit_indices = get_qbit_indices([q.get_identifier() for q in self.qbits], gate.arguments)

                combined_initial_gate = np.matmul(
                    identity_pad_gate(gate.matrix
                                      if not are_qbits_reversed(qbit_indices)
                                      else gate.matrix_swapped,
                                      qbit_indices,
                                      len(self.qbits)),
                    combined_initial_gate)

            self.solver.add(state_equals(state_sequence.states[0],
                                         matrix_vector_multiplication(to_complex_matrix(combined_initial_gate),
                                                                      initial_state_definition)))
        else:
            self.solver.add(state_equals(state_sequence.states[0], initial_state_definition))

        # 3 Iterate all operations and build intermediate states
        for (i, operation) in enumerate(self.program):
            if isinstance(operation, Gate) or isinstance(operation, List):
                if len(state_sequence.measured_states) > 0:
                    raise Exception('Gates after measurement are not supported.')

                # current state vector
                previous_state = state_sequence.states[-1]
                next_state = state_sequence.add_state()

                # build state operation
                if isinstance(operation, Gate) and operation.oracle_value is None:
                    # operation is singular
                    qbit_indices = get_qbit_indices([q.get_identifier() for q in self.qbits], operation.arguments)

                    if not are_qbits_adjacent(qbit_indices):
                        state_operation = swap_transform_non_adjacent_gate(operation.matrix,
                                                                           qbit_indices,
                                                                           len(self.qbits))
                    else:
                        state_operation = identity_pad_gate(operation.matrix
                                                            if not are_qbits_reversed(qbit_indices)
                                                            else operation.matrix_swapped,
                                                            qbit_indices,
                                                            len(self.qbits))
                    # build operation output
                    self.solver.add(state_equals(next_state,
                                                 matrix_vector_multiplication(to_complex_matrix(state_operation),
                                                                              previous_state)))
                elif isinstance(operation, Gate) and operation.oracle_value is not None:
                    self.solver.add(state_equals_phase_oracle(previous_state, next_state, operation.oracle_value))
                else:
                    # operation is composite
                    state_operation = identity_pad_gate(I_matrix, [0], len(self.qbits))

                    for operation_element in operation:
                        qbit_indices = get_qbit_indices([q.get_identifier() for q in self.qbits],
                                                        operation_element.arguments)

                        if not are_qbits_adjacent(qbit_indices):
                            state_operation = swap_transform_non_adjacent_gate(operation_element.matrix,
                                                                               qbit_indices,
                                                                               len(self.qbits))
                        else:
                            state_operation = np.matmul(identity_pad_gate(operation_element.matrix
                                                                          if not are_qbits_reversed(qbit_indices)
                                                                          else operation_element.matrix_swapped,
                                                                          qbit_indices,
                                                                          len(self.qbits)),
                                                        state_operation)

                    # build operation output
                    self.solver.add(state_equals(next_state,
                                                 matrix_vector_multiplication(to_complex_matrix(state_operation),
                                                                              previous_state)))
            elif isinstance(operation, Measurement):
                # Branch out into different states
                previous_state = state_sequence.states[-1]
                exists_measurement_state = len(state_sequence.measured_states) > 0

                if isinstance(operation.arguments, QbitVal):
                    measurement_states = state_sequence.add_measurement_state()

                    qbit_index = get_qbit_indices([q.get_identifier() for q in self.qbits], [operation.arguments])[0]

                    for (j, measurement_state) in enumerate(measurement_states):
                        measurement_operation = identity_pad_gate(zero_measurement
                                                                  if j % 2 == 0
                                                                  else one_measurement,
                                                                  [qbit_index],
                                                                  len(self.qbits))

                        if not exists_measurement_state:
                            # First measured state
                            self.solver.add(
                                state_equals(measurement_state,
                                             matrix_vector_multiplication(to_complex_matrix(measurement_operation),
                                                                          previous_state)))
                        else:
                            # Existing measured states
                            for state_before_element in previous_state:
                                self.solver.add(
                                    state_equals(measurement_state,
                                                 matrix_vector_multiplication(to_complex_matrix(measurement_operation),
                                                                              state_before_element)))
                else:
                    measurement_states = state_sequence.add_measurement_state(len(operation.arguments))

                    qbit_indices = get_qbit_indices([q.get_identifier() for q in self.qbits], operation.arguments)

                    num_digits = len('{0:b}'.format(len(measurement_states))) - 1
                    binary_format = '{0:0' + str(num_digits) + 'b}'

                    if measurement_branch is not None:
                        state_sequence.states[-1] = [measurement_states[measurement_branch]]
                        measurement_states = state_sequence.states[-1]

                    for (j, measurement_state) in enumerate(measurement_states):
                        bit_vector = binary_format.format(j if measurement_branch is None else measurement_branch)

                        measurement_ops = []

                        for b in bit_vector:
                            if b == '0':
                                measurement_ops.append(zero_measurement)
                            else:
                                measurement_ops.append(one_measurement)

                        combined_measurement = kron(measurement_ops)

                        measurement_operation = identity_pad_gate(combined_measurement,
                                                                  qbit_indices,
                                                                  len(self.qbits))

                        if not exists_measurement_state:
                            # First measured state
                            self.solver.add(
                                state_equals(measurement_state,
                                             matrix_vector_multiplication(to_complex_matrix(measurement_operation),
                                                                          previous_state)))
                        else:
                            # Existing measured states
                            raise Exception('No multi-measurement after other measurements.')
            else:
                raise Exception('Unsupported operation. Has to be either gate or measurement.')

        # 4.1 Repair synthesis:
        if self.final_qbits is not None and synthesize_repair is True:
            raise Exception('State model does not support repair')

        # 4.2 Final state qbits
        if self.final_qbits is not None:
            final_state_definition = complex_kron_n_ary([qbit.to_complex_list() for qbit in self.final_qbits])

            if len(state_sequence.measured_states) == 0:
                self.solver.add(state_equals(state_sequence.states[-1], final_state_definition))
            else:
                # build disjunction for the different measurement results
                disjunction_elements = []

                for final_state in state_sequence.states[-1]:
                    disjunction_elements.append(state_equals(final_state, final_state_definition))

                self.solver.add(Or(disjunction_elements))

        # 5 Call solver
        qbit_identifiers = [qbit.get_identifier() for qbit in self.qbits]

        if file_generation_only:
            (temp_file, qbit_identifiers_out) = write_smt_file(self.solver,
                                                               qbit_identifiers,
                                                               state_sequence,
                                                               self.specification,
                                                               self.specification_type,
                                                               self.is_equality_specification,
                                                               output_qbits=[q.get_identifier() for q in
                                                                             self.final_qbits]
                                                               if self.final_qbits is not None else None,
                                                               overapproximation=overapproximation,
                                                               dump_smt_encoding=dump_smt_encoding)

            return temp_file, qbit_identifiers_out
        else:
            (sat_result, model) = solve(self.solver,
                                        qbit_identifiers,
                                        state_sequence,
                                        self.specification,
                                        self.specification_type,
                                        self.is_equality_specification,
                                        output_qbits=[q.get_identifier() for q in self.final_qbits]
                                        if self.final_qbits is not None else None,
                                        delta=self.delta,
                                        overapproximation=overapproximation,
                                        dump_smt_encoding=dump_smt_encoding,
                                        dump_solver_output=dump_solver_output)

            end_full = time.time()
            time_full = end_full - start_full

            print(f'\nElapsed time {precision_format.format(time_full)} seconds.')
            return sat_result, model, time_full


def _has_boolean_gates_only(program):
    for (i, gate) in enumerate(program):
        if gate.name not in ['I', 'X', 'SWAP', 'CNOT', 'CCX']:
            return False

    return True
