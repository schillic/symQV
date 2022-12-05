from typing import Union, Callable, List
from typing import Optional as Opt

from pyparsing import Word, alphas, nums, Optional, SkipTo, alphanums

input_spec_parser = 'q_in' + Optional('[' + Word(nums) + ']') \
                    + '=' + Word(alphas) + ';'

output_spec_parser = 'q_out' + Optional('[' + Word(nums) + ']') \
                     + '=' + ('q_in' + Optional('[' + Word(nums) + ']') |
                              SkipTo(';')) \
                     + ';'

bitvector_parser = SkipTo('q_in') + 'q_in' + '[' + Word(alphanums) + ':' + Word(alphanums) + ']'


class BitVector:
    def __init__(self, start: Union[int, str], end: Union[int, str], function: str):
        """
        A bitvector class used for an output specification.
        :param start: input index where the vector starts.
        :param end: output index (exclusive) where vector ends.
        :param function: a factor with which the bitvector should be multiplied.
        """
        self.start = start
        self.end = end
        self.function = function

    def __str__(self):
        return f"{self.function.replace(' x', f' q_in[{self.start}:{self.end}]')}"


class Equality:
    def __init__(self, left: Union[int, None], right: Union[int, str, BitVector],
                 name_left: Opt[str], name_right: Opt[str]):
        """
        Equality specification.
        :param left: left hand side of the equality.
        :param right: right hand side of the equality.
        :param name_left: left hand side variable's name.
        :param name_right: right hand side variable's name.
        """
        self.left = left
        self.right = right
        self.name_left = name_left
        self.name_right = name_right

    def is_symbolic(self) -> bool:
        """
        :return: True if the specification is symbolic, i.e. the value is not fixed.
        """
        return type(self.right) == str

    def is_all_inputs(self) -> bool:
        """
        :return: True if the equality addresses the complete register.
        """
        return self.left is None

    def is_bitvector(self) -> bool:
        """
        :return: True if the equality compares with a bit vector.
        """
        return isinstance(self.right, BitVector)

    def __str__(self):
        if self.is_symbolic():
            if self.is_all_inputs():
                return f'{self.name_left} = {self.right}'
            else:
                return f'{self.name_left}[{self.left}] = {self.right}'
        elif self.is_bitvector():
            return f'{self.name_left}[{self.left}] = {self.right}'
        else:
            return f'{self.name_left}[{self.left}] = {self.name_right}[{self.right}]'


class Specification:
    def __init__(self, inputs: List[Equality] = None, outputs: List[Equality] = None):
        """
        Input - Output specification.
        :param inputs: input equality specification.
        :param outputs: output equality specification.
        """
        if outputs is None:
            outputs = []
        if inputs is None:
            inputs = []

        self.inputs = inputs
        self.outputs = outputs

    def __str__(self):
        return '\n'.join([f'{input};' for input in self.inputs]) + '\n' \
               + '\n'.join([f'{output};' for output in self.outputs])

    @staticmethod
    def read_from_file(path: str):
        """
        Read in a specification file.
        :param path: path to file.
        :return: specification object.
        """
        specification = Specification()

        with open(path) as file:
            lines = file.readlines()

        for line in lines:
            if line.startswith('//') or line.startswith('QSPEC'):
                continue
            elif line.startswith('q_in'):
                parsed = input_spec_parser.parseString(line)

                if parsed[1] == '[':
                    input_spec = Equality(left=int(parsed[2]),
                                          right=parsed[5],
                                          name_left='q_in',
                                          name_right=None)
                else:
                    input_spec = Equality(left=None,
                                          right=parsed[2],
                                          name_left='q_in',
                                          name_right=None)

                specification.inputs.append(input_spec)
            elif line.startswith('q_out'):
                parsed = output_spec_parser.parseString(line)

                if parsed[5] == 'q_in':
                    output_spec = Equality(left=int(parsed[2]),
                                           right=int(parsed[7]),
                                           name_left='q_out',
                                           name_right='q_in')
                else:
                    parsed_bitvector = bitvector_parser.parseString(line)

                    start = int(parsed_bitvector[3])
                    end = int(parsed_bitvector[5])
                    function = parsed[5].replace(f' q_in[{start}:{end}]', ' x')

                    bitvector = BitVector(start, end, function)
                    output_spec = Equality(left=int(parsed[2]),
                                           right=bitvector,
                                           name_left='q_out',
                                           name_right='q_in')

                specification.outputs.append(output_spec)
            else:
                raise ValueError(f'Could not parse line: {line}')

        return specification
