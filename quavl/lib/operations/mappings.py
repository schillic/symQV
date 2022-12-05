from typing import List

import numpy as np

from quavl.lib.expressions.complex import ComplexVal
from quavl.lib.expressions.qbit import QbitVal


def X_mapping(qbit: QbitVal) -> QbitVal:
    return QbitVal(qbit.beta, qbit.alpha)


def H_mapping(qbit: QbitVal) -> QbitVal:
    return QbitVal((qbit.alpha + qbit.beta).__rdiv__(np.sqrt(2)),
                   (qbit.alpha - qbit.beta).__rdiv__(np.sqrt(2)))


def CNOT_mapping(tensor: List[ComplexVal]) -> List[ComplexVal]:
    if len(tensor) != 4:
        raise Exception(f'Tensor needs to be length 4, was {len(tensor)}.')

    return [tensor[0], tensor[1], tensor[3], tensor[2]]


def CNOT_H_mapping(tensor: List[ComplexVal]) -> List[ComplexVal]:
    if len(tensor) != 4:
        raise Exception(f'Tensor needs to be length 4, was {len(tensor)}.')

    return [(tensor[0] + tensor[3]).__rdiv__(np.sqrt(2)),
            (tensor[1] + tensor[2]).__rdiv__(np.sqrt(2)),
            (tensor[0] - tensor[3]).__rdiv__(np.sqrt(2)),
            (tensor[1] - tensor[2]).__rdiv__(np.sqrt(2))]
