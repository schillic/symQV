from enum import Enum
from typing import List, Tuple

import numpy as np

from symqv.lib.constants import I_matrix
from symqv.lib.operations.measurements import zero_measurement, one_measurement, \
    get_measurement_probability_from_state
from symqv.lib.utils.arithmetic import kron
from symqv.lib.utils.helpers import identity_pad_gate
from symqv.lib.utils.state_analysis import normalize, are_qbits_consecutive, swap_arguments_to_front


class Condition:
    def __init__(self, matches_value: int):
        """
        Condition based on a classical bit and a value to compare for equality.
        :param matches_value: 0 or 1.
        """
        self.matches_value = matches_value


class OperationType(Enum):
    gate = 'gate'
    measurement = 'measurement'


class Operation:
    def __init__(self, type: OperationType,
                 name: str,
                 num_qbits: int,
                 inputs: List[int],
                 output: int = None,
                 matrix: np.ndarray = None,
                 definition: List[float] = None,
                 condition: Condition = None):
        """
        Initialize a quantum operation.
        :param type: gate or measurement.
        :param name: name of the operation.
        :param num_qbits: number of qbits in circuit.
        :param inputs: input qbit indices.
        :param output: output cbit index (in case of measurement).
        :param matrix: gate unitary (optional).
        :param definition: if the gate is given explicitely (optional).
        :param condition: execute gate on cbit equality condition (optional).
        """
        self.type = type
        self.name = name
        self.num_qbits = num_qbits
        self.inputs = inputs
        self.output = output
        self.matrix = matrix
        self.definition = definition
        self.condition = condition

    def __str__(self):
        if self.type == OperationType.gate:
            return (f'if(c=={self.condition.matches_value}) '
                    if self.condition is not None else '') \
                   + f'{self.name}' \
                   + (('(' + ','.join(str(v) for v in self.definition) + ') ')
                      if self.definition is not None else ' ') \
                   + ','.join([f'q[{i}]' for i in [-(v - (self.num_qbits - 1)) for v in self.inputs]]) \
                   + ';'
        if self.type == OperationType.measurement:
            return f'measure q[{self.inputs[0]}] -> c[{self.output}];'


class HybridCircuit:
    def __init__(self, qbits: int, cbits: int):
        """
        Initialize a hybrid quantum circuit.
        :param qbits: number of quantum bits.
        :param cbits: number of classical bits.
        """
        self.qbits = qbits
        self.cbits = cbits
        self.operations = []
        self.gate_declarations = []

    def add(self, operation: Operation):
        """
        Add an operation to the circuit.
        :param operation: operation.
        """
        self.operations.append(operation)

    def get_unitaries(self) -> List[np.ndarray]:
        """
        Get unitaries for each unitary computation sequence, i.e. the unitary between each pair of
        measurements.
        :return: List of unitaries.
        """
        unitaries = []

        current_unitary = kron([I_matrix for _ in range(self.qbits)])

        for op in self.operations:
            if op.type == OperationType.gate:
                if op.matrix is None:
                    raise Exception('No matrix: ', op)

                padded_op = identity_pad_gate(op.matrix, op.inputs, self.qbits)

                if op.condition is not None:
                    unitaries.append(current_unitary)
                    current_unitary = kron([I_matrix for _ in range(self.qbits)])

                current_unitary = np.matmul(padded_op, current_unitary)

        return unitaries

    def get_final_states(self, initial_state: np.array) -> List[np.array]:
        """
        Transform a given initial state to final state(s).
        :param initial_state: vector of the initial state.
        :return: list of final state vectors.
        """
        return self._get_final_states(initial_state)[0]

    def _get_final_states(self, initial_state: np.array) -> Tuple[List[np.array], List[np.array]]:
        """
        Transform a given initial state to final state(s).
        :param initial_state: vector of the initial state.
        :return: list of final state vectors and list of classical state vectors.
        """
        if len(initial_state) != 2 ** self.qbits:
            raise ValueError(f'Initial state vector should have {2 ** self.qbits} entries, '
                             f'but had {len(initial_state)}.')

        # these hold multiple states because of branching
        current_q_states = [initial_state]
        current_c_states = [np.zeros(self.cbits)]

        for op in self.operations:
            if op.type == OperationType.gate:
                swap = None

                if are_qbits_consecutive(op.inputs):
                    # no swap required if arguments are consecutive
                    padded_op = identity_pad_gate(op.matrix, op.inputs, self.qbits)
                else:
                    # swap qbits together if not consecutive
                    swap_operation, swap_indices = swap_arguments_to_front(op.inputs, self.qbits)

                    swap = identity_pad_gate(swap_operation,
                                             swap_indices,
                                             self.qbits)

                    # swap
                    current_q_states = [np.matmul(swap, s) for s in current_q_states]

                    # apply gate
                    padded_op = identity_pad_gate(op.matrix,
                                                  list(range(min(op.inputs), min(op.inputs) + len(op.inputs))),
                                                  self.qbits)

                if op.condition is None:
                    # standard gate application
                    current_q_states = [np.matmul(padded_op, s) for s in current_q_states]
                else:
                    # if condition
                    current_c_int_states = [(''.join([str(int(bit)) for bit in c_state]))
                                            for c_state in current_c_states]

                    current_q_states = [np.matmul(padded_op, state)
                                        if int(current_c_int_states[i], 2) == op.condition
                                        else state
                                        for i, state in enumerate(current_q_states)]

                if swap is not None:
                    # swap back
                    current_q_states = [np.matmul(swap.T.conj(), s) for s in current_q_states]

            elif op.type == OperationType.measurement:
                # Two possible measurement outcomes results in two branches per measurement.
                zero = zero_measurement
                one = one_measurement

                new_q_states = []
                new_c_states = []

                for i, state in enumerate(current_q_states):
                    # For cases where the qbits are in basis states, there is only one branch.
                    # These cases have to be identified because to remove impossible measurement outcomes.
                    zero_prob = get_measurement_probability_from_state(op.inputs[0], state)

                    if np.isclose(zero_prob, 1):
                        # 0 measurement
                        new_q_states.append(normalize(np.matmul(identity_pad_gate(zero, op.inputs, self.qbits), state)))
                        new_c_states.append(np.array([0 if j == op.inputs[0] else current_c_states[i][j]
                                                      for j in range(self.qbits)]))
                    elif np.isclose(zero_prob, 0):
                        # 1 measurement
                        new_q_states.append(normalize(np.matmul(identity_pad_gate(one, op.inputs, self.qbits), state)))
                        new_c_states.append(np.array([1 if j == op.inputs[0] else current_c_states[i][j]
                                                      for j in range(self.qbits)]))
                    else:
                        # both outcomes possible
                        new_q_states.append(normalize(np.matmul(identity_pad_gate(zero, op.inputs, self.qbits), state)))
                        new_q_states.append(normalize(np.matmul(identity_pad_gate(one, op.inputs, self.qbits), state)))

                        new_c_states.append(np.array([0 if j == op.inputs[0] else current_c_states[i][j]
                                                      for j in range(self.qbits)]))
                        new_c_states.append(np.array([1 if j == op.inputs[0] else current_c_states[i][j]
                                                      for j in range(self.qbits)]))

                current_q_states = new_q_states
                current_c_states = new_c_states

        return current_q_states, current_c_states

    def __str__(self):
        """
        :return: QASM representation of the circuit.
        """
        return f'qreg q[{self.qbits}];\n' \
               + (('\n'.join([f'gate {v} q ' + '{ }' for v in self.gate_declarations]))
                  if self.gate_declarations is not None else '') \
               + (f'creg c[{self.cbits}];\n'
                  if self.cbits is not None else '') \
               + '\n'.join([str(op) for op in self.operations])
