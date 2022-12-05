from typing import List, Optional, Union

from numpy.core.multiarray import ndarray

from quavl.lib.expressions.qbit import QbitVal


class Gate:
    def __init__(self,
                 name: str,
                 arguments: List[QbitVal],
                 matrix: Optional[ndarray],
                 matrix_swapped: ndarray = None,
                 mapping = None,
                 r_mapping = None,
                 parameter = None,
                 oracle_value: int = None):
        self.name = name
        self.arguments = arguments
        self.matrix = matrix
        self.matrix_swapped = matrix_swapped
        self.mapping = mapping
        self.r_mapping = r_mapping
        self.parameter = parameter
        self.control_qbits = None
        self.oracle_value = oracle_value

    def __pow__(self, power, modulo=None):
        """
        Power of gate.
        :param power: power.
        :param modulo: power of modulo (not supported).
        :return: power of gate.
        """
        if modulo is not None:
            raise Exception('Power modulo is not supported')

        return Gate(f'self.name ** {power}',
                    self.arguments,
                    self.matrix ** power,

                    None if self.matrix_swapped is None else self.matrix_swapped ** power)

    def __repr__(self):
        if self.oracle_value is None:
            string = self.name

            if self.parameter is not None:
                string += f'_{self.parameter}'

            if isinstance(self.arguments[0], list):
                string += f'({self.arguments})'
            else:
                string += f'({", ".join([q.get_identifier() for q in self.arguments])})'

            if self.control_qbits is not None:
                string += f'.controlled_by({", ".join([q.get_identifier() for q in self.control_qbits])})'

            return string
        else:
            return f'{self.name}_{self.oracle_value}({", ".join([q.get_identifier() for q in self.arguments])})'

    def controlled_by(self, control_qbits: Union[QbitVal, List[QbitVal]]):
        if isinstance(control_qbits, QbitVal):
            self.control_qbits = [control_qbits]
        else:
            self.control_qbits = control_qbits
        return self

    def arity(self):
        return len(self.arguments)
