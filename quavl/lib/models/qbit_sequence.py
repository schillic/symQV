from typing import List, Union, Optional

from quavl.lib.expressions.qbit import QbitVal, Qbits
from quavl.lib.expressions.rqbit import RQbitVal, RQbits


class QbitSequence:
    def __init__(self, qbits: List[QbitVal]):
        self.num_states = 1
        self.qbits = qbits
        self.num_qbits = len(qbits)
        self.states = [self.build_initial_state()]
        self.reduced_state_space = isinstance(qbits[0], RQbitVal)
        self.measured_states = []
        self.has_multi_measurement = False

    def build_initial_state(self) -> List[Union[QbitVal, RQbitVal]]:
        """
        Build initial state from qbits.
        :param qbits: qbits.
        :return: Initial state.
        """
        return self.qbits

    def add_state(self, suffix: Optional[str] = None) -> List[Union[QbitVal, RQbitVal]]:
        """
        Add an additional state.
        :param suffix: an optional suffix. Leads to errors if the same suffix is used more than once.
        """
        if suffix is None:
            suffix = str(self.num_states)

        if self.reduced_state_space:
            new_state = RQbits([f'{q.get_identifier()}_{suffix}' for q in self.qbits])
        else:
            new_state = Qbits([f'{q.get_identifier()}_{suffix}' for q in self.qbits])

        self.states.append(new_state)

        self.num_states += 1
        return new_state


    def __str__(self):
        return f'StateSequence(num_states={self.num_states}, num_qbits={self.num_qbits})'
