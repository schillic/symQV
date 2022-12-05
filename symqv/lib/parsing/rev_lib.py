from typing import List

from z3 import And, Or

from symqv.lib.expressions.rqbit import RQbits, RQbitVal
from symqv.lib.models.circuit import Circuit
from symqv.lib.models.gate import Gate
from symqv.lib.operations.gates import X, V, V_dag, Peres, Peres_inv


def to_specification(path: str, input_qbits: List[RQbitVal], output_qbits: List[RQbitVal]):
    num_input_vars = None
    num_output_vars = None

    specification_elements = []

    with open(path) as file:
        lines = file.readlines()

    for line in lines:
        line = line.replace('\n', '')

        if line.isspace():
            continue

        if line.startswith('#'):
            continue

        if line.startswith('.i '):
            num_input_vars = int(line.split('.i ')[1])
            continue

        if line.startswith('.o '):
            num_output_vars = int(line.split('.o ')[1])

            if num_input_vars != num_output_vars:
                raise Exception(f'{num_input_vars} inputs, but {num_output_vars} outputs.')

            continue

        if line.startswith('.ilb '):
            variables = line.split('.ilb ')[1].split(' ')
            inputs = [var for var in variables if not (var.isspace() or var == '')]

            if len(inputs) != num_input_vars:
                raise Exception(f'Declared {num_input_vars}, but got {len(inputs)} inputs.')

            continue

        if line.startswith('.ob '):
            variables = line.split('.ob ')[1].split(' ')
            outputs = [var for var in variables if not (var.isspace() or var == '')]

            if len(outputs) != num_output_vars:
                raise Exception(f'Declared {num_output_vars}, but got {len(outputs)} inputs.')

            continue

        if line.startswith('.e'):
            return Or(specification_elements)

        specification_elements.append(_parse_equality(line, input_qbits, output_qbits))


def to_circuit(path: str) -> Circuit:
    """
    Parse file in RevLib format (http://www.informatik.uni-bremen.de/rev_lib) to Circuit.
    :param path: file path.
    :return: Circuit.
    """
    num_vars = None
    qbits = None
    qbit_dict = {}

    inputs = None
    qbit_initial_values = []

    program: List[Gate] = []

    is_define_env = False

    with open(path) as file:
        lines = file.readlines()

    for line in lines:
        line = line.replace('\n', '')

        if line.isspace():
            continue

        if line.startswith('#'):
            continue

        if line.startswith('.enddefine'):
            is_define_env = False
            continue

        if is_define_env or line.startswith('.define'):
            is_define_env = True
            continue

        if line.startswith('.version'):
            continue

        if line.startswith('.numvars'):
            num_vars = int(line.split('.numvars ')[1])
            continue

        if line.startswith('.variables'):
            variables = line.split('.variables ')[1].split(' ')
            variables = [var for var in variables if not (var.isspace() or var == '')]

            qbits = RQbits(variables)

            if len(qbits) != num_vars:
                raise Exception(f'Declared {num_vars} variables, but got {len(qbits)} qbits.')

            for i, qbit in enumerate(qbits):
                qbit_dict[variables[i]] = qbit
            continue

        if line.startswith('.inputs'):
            inputs = line.split('.inputs ')[1].split(' ')

            inputs = [input for input in inputs if not (input.isspace() or input == '')]

            for i, input in enumerate(inputs):
                if input.isnumeric():
                    value = int(input)
                    if value == 0:
                        qbit_initial_values.append((1, 0))
                    elif value == 1:
                        qbit_initial_values.append((0, 1))
                    else:
                        raise Exception(f'Unsupported initial value {value} at {i}.')
                elif input.isalpha() or input.isalnum():
                    qbit_initial_values.append(None)
                else:
                    raise Exception(f'Unsupported initial input {input} at {i}.')
            continue

        if line.startswith('.outputs'):
            outputs = line.split('.outputs ')[1].split(' ')

            outputs = [output for output in outputs if not (output.isspace() or output == '')]

            if len(outputs) != len(inputs):
                raise Exception(f'Declared {len(outputs)} outputs, but has {len(inputs)} inputs.')
            continue

        if line.startswith('.constants'):
            constants = line.split('.constants ')[1]

            for i, flag in enumerate(constants):
                if flag == '-' and qbit_initial_values[i] is not None \
                        or flag == '0' and qbit_initial_values[i] != (1, 0) \
                        or flag == '1' and qbit_initial_values[i] != (0, 1):
                    raise Exception(f'Constant flag at {i} was {flag}, but input was {qbit_initial_values[i]}.')
            continue

        if line.startswith('.garbage'):
            garbage = line.split('.garbage ')[1]

            for i, flag in enumerate(garbage):
                if flag == '1' and qbit_initial_values[i] is not None:
                    raise Exception(f'Garbage flag at {i} was {flag}, but input was {qbit_initial_values[i]}.')
            continue

        if line.startswith('.begin'):
            continue

        if line.startswith('.end'):
            circuit = Circuit(qbits, program)

            if not all(v is None for v in qbit_initial_values):
                circuit.initialize(qbit_initial_values)

            return circuit

        instruction = _parse_instruction(line, qbit_dict)

        if instruction.name == 'Peres':
            program.append(X(instruction.arguments[2]).controlled_by(instruction.arguments[:2]))
            program.append(X(instruction.arguments[1]).controlled_by(instruction.arguments[:1]))
        elif instruction.name == 'Peres_inv':
            program.append(X(instruction.arguments[1]).controlled_by(instruction.arguments[:1]))
            program.append(X(instruction.arguments[2]).controlled_by(instruction.arguments[:2]))
        else:
            program.append(instruction)


def _parse_equality(line, input_qbits: List[RQbitVal], output_qbits: List[RQbitVal]):
    """
    Parse equality line from .pla format.
    :param line: string.
    :param input_qbits: input RQbitVals.
    :param output_qbits: outputs RQbitVals.
    :return:
    """
    [input_values, output_values] = line.split()

    conjunction_elements = []

    for i in range(len(input_values)):
        conjunction_elements.append(input_qbits[i].z0 == (not bool(int(input_values[i]))))
        conjunction_elements.append(input_qbits[i].z1 == bool(int(input_values[i])))

    for i in range(len(output_values)):
        conjunction_elements.append(output_qbits[i].z0 == (not bool(int(output_values[i]))))
        conjunction_elements.append(output_qbits[i].z1 == bool(int(output_values[i])))

    return And(conjunction_elements)


def _parse_instruction(instruction: str, qbit_dict) -> Gate:
    """
    Parse quantum instruction string.
    :param instruction: instruction string.
    :param qbit_dict: dictionary of qbit name QbitVal mappings.
    :return: gate application.
    """
    elements = instruction.split(' ')

    gate_name = elements[0]
    gate = _parse_gate(gate_name)

    parameter_names = elements[1:]
    parameters = [qbit_dict[p] for p in parameter_names]

    if len(parameters) == 1:
        return gate(parameters[0])

    return gate(*parameters)


def _parse_gate(gate_name):
    """
    Parse gate name to gate.
    :param gate_name: gate's name.
    :return: actual gate.
    """
    if gate_name == 't1':
        return X
    if gate_name == 't2':
        return lambda *qbits: X(qbits[1]).controlled_by(qbits[0])
    if gate_name == 't3':
        return lambda *qbits: X(qbits[2]).controlled_by(qbits[0:2])
    if gate_name.startswith('t') and gate_name[1:].isnumeric():
        return lambda *qbits: X(qbits[-1]).controlled_by(qbits[:-1])
    if gate_name == 'v':
        return lambda *qbits: V(qbits[-1]).controlled_by(qbits[:-1])
    if gate_name == 'v+':
        return lambda *qbits: V_dag(qbits[-1]).controlled_by(qbits[:-1])
    if gate_name == 'p':
        return lambda *qbits: Peres(qbits[0], qbits[1], qbits[2])
    if gate_name == 'pi':
        return lambda *qbits: Peres_inv(qbits[0], qbits[1], qbits[2])
    else:
        raise Exception(f'Gate {gate_name} cannot be parsed.')
