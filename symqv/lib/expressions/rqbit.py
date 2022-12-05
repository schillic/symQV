from typing import List

import numpy as np
from z3 import Bool, BoolRef, And, Or

from symqv.lib.expressions.qbit import QbitVal


class RQbitVal(QbitVal):
    def __init__(self, z0: BoolRef = False, z1: BoolRef = False,
                 h0: BoolRef = False, h1: BoolRef = False,
                 zm0: BoolRef = False, zm1: BoolRef = False,
                 hm0: BoolRef = False, hm1: BoolRef = False,
                 v0: BoolRef = False, v1: BoolRef = False):
        """
        Constructor for a reduced state space qbit.
        :param z0: (1, 0) (Computational basis)
        :param z1: (0, 1)
        :param h0: (1, 1)/sqrt(2) (Hadamard basis)
        :param h1: (1, -1)/sqrt(2)
        :param zm0: (-1, 0)
        :param zm1: (0, -1)
        :param hm0: (-1, -1)/sqrt(2)
        :param hm1: (-1, 1)/sqrt(2)
        :param v0: ((1+1j)/2, (1-1j)/2)
        :param v1: ((1-1j)/2, (1+1j/2)
        """
        super().__init__()

        if not isinstance(z0, BoolRef) and sum(np.array([z0, z1, h0, h1, zm0, zm1, hm0, hm1, v0, v1], dtype=int)) != 1:
            raise ValueError('Exactly one parameter has to be one.')

        self.z0 = z0
        self.z1 = z1

        self.h0 = h0
        self.h1 = h1

        self.zm0 = zm0
        self.zm1 = zm1

        self.hm0 = hm0
        self.hm1 = hm1

        self.v0 = v0
        self.v1 = v1

    def __eq__(self, other: 'RQbitVal'):
        return self.z0 == other.z0 and self.z1 == other.z1 and \
               self.h0 == other.h0 and self.h1 == other.h1 and \
               self.zm0 == other.zm0 and self.zm1 == other.zm1 and \
               self.hm0 == other.hm0 and self.hm1 == other.hm1 and \
               self.v0 == other.v0 and self.v1 == other.v1

    def __neq__(self, other: 'RQbitVal'):
        return self.z0 != other.z0 or self.z1 != other.z1 or \
               self.h0 != other.h0 or self.h1 != other.h1 or \
               self.zm0 != other.zm0 or self.zm1 != other.zm1 or \
               self.hm0 != other.hm0 or self.hm1 != other.hm1 or \
               self.v0 != other.v0 or self.v1 != other.v1

    def __neg__(self):
        return RQbitVal(z0=self.z1, z1=self.z0,
                        h0=self.h1, h1=self.h0,
                        zm0=self.zm1, zm1=self.zm0,
                        hm0=self.hm1, hm1=self.hm0,
                        v0=self.v1, v1=self.v0)

    def __repr__(self):
        return self.get_identifier()

    def get_constraints(self, computational_basis_only: False):
        if computational_basis_only:
            attributes = ['z0', 'z1']
        else:
            attributes = ['z0', 'z1', 'v0', 'v1']  # ['z0', 'z1', 'h0', 'h1', 'zm0', 'zm1', 'hm0', 'hm1', 'v0', 'v1']

        conjunctions = []
        for attribute in attributes:
            conjunction = And([getattr(self, v) == (v == attribute) for v in attributes])
            conjunctions.append(conjunction)

        return Or(conjunctions)

    def get_identifier(self):
        """
        :return: Identifier.
        """
        return str(self.z0).split('.')[0]

    def to_complex_list(self):
        """
        Unfold an rqbit into a 2-element list of complex values.
        :return: 2-element list of complex values.
        """
        return [self.z0, self.z1]


def RQbit(identifier: str) -> RQbitVal:
    """
    Generate a named rqbit.
    :param identifier: chosen identifier.
    :return: RQbit.
    """
    return RQbitVal(Bool('%s.z0' % identifier), Bool('%s.z1' % identifier),
                    Bool('%s.h0' % identifier), Bool('%s.h1' % identifier),
                    Bool('%s.zm0' % identifier), Bool('%s.zm1' % identifier),
                    Bool('%s.hm0' % identifier), Bool('%s.hm1' % identifier),
                    Bool('%s.v0' % identifier), Bool('%s.v1' % identifier))


def RQbits(identifiers: List[str]) -> List[RQbitVal]:
    """
    Generate many named rqbits.
    :param identifiers: chose identifiers.
    :return: List of rqbits.
    """
    return [RQbit(identifier) for identifier in identifiers]
