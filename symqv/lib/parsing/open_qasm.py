from typing import List

import numpy as np
from pyparsing import Word, Optional, Combine, alphas, alphanums, nums, delimitedList

from symqv.lib.constants import H_matrix, X_matrix, Y_matrix, Z_matrix, U3_matrix, CZ_matrix, \
    CNOT_matrix, CU1_matrix, CRk_matrix
from symqv.lib.models.hybrid_circuit import HybridCircuit, Operation, OperationType, Condition

# parser templates
q_register = 'qreg' + Word(alphanums) + '[' + Word(nums) + '];'
c_register = 'creg' + Word(alphanums) + '[' + Word(nums) + '];'

decimal = Combine(Optional('-') + Word(nums) + Optional('.' + Word(nums, excludeChars=[','])))
rational = Combine(Word(alphanums) + '/' + Word(alphanums))

gate_parser = Word(alphanums) \
              + Optional('(' + delimitedList(decimal | rational, delim=',') + ')') \
              + delimitedList(Word(alphanums) + '[' + Word(nums) + ']', delim=',') + ';'

gate_decl_parser = 'gate' + Word(alphas) + Word(alphas) + '{' + '}'

measurement_parser = 'measure' + Word(alphanums) + Optional('[' + Word(nums) + ']') \
                     + '->' + Word(alphanums) + Optional('[' + Word(nums) + ']') + ';'

condition_parser = 'if' + '(' + Word(alphas) + '==' + Word(nums) + ')' \
                   + Word(alphanums) + Word(alphanums) + '[' + Word(nums) + ']' + ';'

qasm_gate_keys = [
    'u3', 'u2', 'u1', 'cx', 'id', 'u0', 'u', 'p', 'x', 'y', 'z', 'h', 's', 'sdg', 't', 'tdg', 'rx', 'ry', 'rz', 'sx',
    'sxdg', 'cz', 'cy', 'swap', 'ch', 'ccx', 'cswap', 'crx', 'cry', 'crz', 'cu1', 'cp', 'cu3', 'csx', 'cu', 'rxx',
    'rzz', 'rccx', 'rc3x', 'c3x', 'c3sqrtx', 'c4x', 'q', 'c']


def read_from_qasm(path: str) -> HybridCircuit:
    """
    Read a .qasm file into a hybrid circuit object.
    :param path: path.
    :return: hybrid circuit.
    """
    hybrid_circuit = None

    num_qbits = None
    name_qreg = None

    name_creg = None

    gate_declarations = []

    with open(path) as file:
        lines = file.readlines()

    for line in lines:
        line = line.replace('\n', '')

        if line.isspace() or line.startswith('//') \
                or line.startswith('OPENQASM') \
                or line.startswith('include') \
                or line.startswith('barrier'):
            continue
        elif line.startswith('qreg'):
            # Case: qbit register declaration
            parsed = q_register.parseString(line)
            name_qreg = parsed[1]
            num_qbits = int(parsed[3])
        elif line.startswith('creg'):
            # Case: cbit register declaration
            if name_creg is None:
                parsed = c_register.parseString(line)
                name_creg = parsed[1]
                num_cbits = int(parsed[3])

                hybrid_circuit = HybridCircuit(num_qbits, num_cbits)
            else:
                raise ValueError('Only one classical register is supported.')
        elif line.startswith('gate'):
            # Case: gate declaration
            parsed = gate_decl_parser.parseString(line)
            gate_declarations.append(parsed[1])
        elif line.startswith('measure'):
            # Case: measurement
            parsed = measurement_parser.parseString(line)

            if parsed[2] == '->':
                # measure all qbits to all cbits
                for qbit in range(num_qbits):
                    hybrid_circuit.add(Operation(type=OperationType.measurement,
                                                 name='measure',
                                                 num_qbits=num_qbits,
                                                 inputs=[qbit],
                                                 output=qbit))
            else:
                # measure certain qbits
                m_input = int(parsed[3])
                m_output = int(parsed[8])

                hybrid_circuit.add(Operation(type=OperationType.measurement,
                                             name='measure',
                                             num_qbits=num_qbits,
                                             inputs=[-m_input + (num_qbits - 1)],
                                             output=-m_output + (num_qbits - 1)))
        elif line.startswith('if'):
            # Case: condition
            parsed = condition_parser.parseString(line)

            gate = to_gate(parsed[5:], num_qbits, name_qreg)
            gate.condition = Condition(int(parsed[3]))

            hybrid_circuit.add(gate)
        else:
            # Case: Gate
            parsed = gate_parser.parseString(line)

            hybrid_circuit.add(to_gate(parsed, num_qbits, name_qreg))

    hybrid_circuit.gate_declarations = gate_declarations

    return hybrid_circuit


def to_gate(parsed: List, num_qbits: int, name_qreg: str) -> Operation:
    """
    Gate from parsed line, optionally with definition.
    :param parsed: parsed line (list).
    :param num_qbits: number of qbits.
    :param name_qreg: name of the qbit register.
    :return: operation of type gate.
    """
    inputs = []
    inside_gate_definition = False
    gate_definition = []

    for i in range(len(parsed)):
        if parsed[i] == '(':
            inside_gate_definition = True
        elif inside_gate_definition:
            if parsed[i] == ')':
                inside_gate_definition = False
                continue

            # check if float
            try:
                gate_definition.append(float(parsed[i]))
            except ValueError:
                if 'pi/' in parsed[i]:
                    denominator = float(parsed[i].split('/')[1])
                    gate_definition.append(np.pi / denominator)
                else:
                    raise Exception(f'Gate parameter {parsed[i]} not supported.')

        if not inside_gate_definition and parsed[i] == name_qreg:
            inputs.append(int(parsed[i + 2]))

    return Operation(type=OperationType.gate,
                     name=parsed[0],
                     num_qbits=num_qbits,
                     inputs=[-v + (num_qbits - 1) for v in inputs],
                     matrix=_to_gate_matrix(parsed[0], None if gate_definition == [] else gate_definition),
                     definition=None if gate_definition == [] else gate_definition)


def _to_gate_matrix(name: str, args: List = None) -> np.ndarray:
    """
    Parse gate name to matrix representation.
    :param name: gate name.
    :param args: optional arguments.
    :return: matrix.
    """
    if name == 'x':
        return X_matrix
    if name == 'y':
        return Y_matrix
    if name == 'z':
        return Z_matrix
    if name == 'h':
        return H_matrix
    if name == 'cx':
        return CNOT_matrix
    if name == 'cz':
        return CZ_matrix
    if name == 'u3':
        return U3_matrix(*args)
    if name == 'cu1':
        return CU1_matrix(args[0])
    if name == 'crk':
        return CRk_matrix(args[0])
    raise Exception(f'Gate {name} not supported.')
