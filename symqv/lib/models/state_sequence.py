from typing import List, Union, Optional

from symqv.lib.expressions.complex import Complexes, ComplexVal
from symqv.lib.expressions.qbit import QbitVal


class StateSequence:
    def __init__(self, qbits: List[QbitVal]):
        self.num_states = 1
        self.qbits = qbits
        self.num_qbits = len(qbits)
        self.states = [self.build_initial_state()]
        self.measured_states = []
        self.has_multi_measurement = False
        self.reduced_state_space = False

    def build_initial_state(self) -> List[ComplexVal]:
        """
        Build initial state from qbits.
        :param qbits: qbits.
        :return: Initial state.
        """
        return Complexes([f'psi_0_{i}' for i in range(2 ** self.num_qbits)])

    def add_state(self, suffix: Optional[str] = None) -> Union[List[ComplexVal], List[List[ComplexVal]]]:
        """
        Add an additional state.
        """
        if suffix is None:
            suffix = str(self.num_states)

        if len(self.measured_states) == 0:
            # No measurements
            new_state = Complexes([f'psi_{suffix}_{i}' for i in range(2 ** self.num_qbits)])

            self.states.append(new_state)

            self.num_states += 1
            return new_state
        else:
            # Existing measurements
            if self.has_multi_measurement:
                raise Exception('No operations possible after multi-measurement')

            num_subsequent_branches = 2 ** len(self.measured_states)
            num_digits = len('{0:b}'.format(num_subsequent_branches)) - 1
            binary_format = '{0:0' + str(num_digits) + 'b}'

            new_state = []

            for i in num_subsequent_branches:
                binary_branch = binary_format.format(i)

                new_state.append(
                    Complexes([f'psi_{suffix}_{j}-{binary_branch}' for j in range(2 ** self.num_qbits)]))

            self.states.append(new_state)

            self.num_states += 1
            return new_state

    def add_measurement_state(self, count: int = 1) -> List[List[ComplexVal]]:
        """
        Add a (list of) measurement state(s).
        :return: The list of measurement states.
        """
        self.measured_states.append(self.num_states)

        if self.has_multi_measurement:
            raise Exception('No operations possible after multi-measurement')

        if count > 1:
            self.has_multi_measurement = True

        num_subsequent_branches = (2 ** count) ** len(self.measured_states)
        num_digits = len('{0:b}'.format(num_subsequent_branches)) - 1
        binary_format = '{0:0' + str(num_digits) + 'b}'

        new_state = []

        for i in range(num_subsequent_branches):
            binary_branch = binary_format.format(i)

            new_state.append(
                Complexes([f'psi_{self.num_states}_{j}-{binary_branch}' for j in range(2 ** self.num_qbits)]))

        self.states.append(new_state)

        self.num_states += 1
        return new_state

    def __str__(self):
        return f'StateSequence(num_states={self.num_states}, num_qbits={self.num_qbits})'
