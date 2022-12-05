from typing import List

from z3 import Real, And, Not

from symqv.lib.constants import cos, sin
from symqv.lib.expressions.complex import Complex, ComplexVal
from symqv.lib.globals import precision_format


class QbitVal:
    def __init__(self, alpha: ComplexVal = None, beta: ComplexVal = None, phi: Real = None, theta: Real = None):
        """
        Constructor for a qbit.
        :param alpha: first magnitude.
        :param beta: second magnitude.
        :param phi: second phase.
        :param theta: third phase.
        """
        self.alpha = alpha
        self.beta = beta

        self.phi = phi
        self.theta = theta

    def __eq__(self, other: 'QbitVal'):
        """
        Equality with a qbit.
        :param other: other qbit.
        :return: Equality.
        """
        if self.alpha is not None and other.alpha is None:
            return And(self.alpha.r == cos(other.theta / 2),
                       self.alpha.i == 0,
                       self.beta.r == cos(other.phi) * sin(other.theta / 2),
                       self.beta.i == sin(other.phi) * sin(other.theta / 2))
        if self.alpha is None and other.alpha is None:
            return And(self.phi == other.phi, self.theta == other.theta)
        return And(self.alpha.r == other.alpha.r, self.alpha.i == other.alpha.i,
                   self.beta.r == other.beta.r, self.beta.i == other.beta.i)

    def isclose(self, other: 'QbitVal', delta: float):
        """
        Equality with a qbit with tolerance.
        :param other: other qbit.
        :param delta: error rate (absolute).
        :return: Equality with tolerance.
        """
        return And(other.alpha.r - delta <= self.alpha.r, self.alpha.r <= other.alpha.r + delta,
                   other.alpha.i - delta <= self.alpha.i, self.alpha.i <= other.alpha.i + delta,
                   other.beta.r - delta <= self.beta.r, self.beta.r <= other.beta.r + delta,
                   other.beta.i - delta <= self.beta.i, self.beta.i <= other.beta.i + delta)

    def __neq__(self, other: 'QbitVal'):
        """
        Unequality with a qbit.
        :param other: other qbit.
        :return: Unequality.
        """
        return And(Not(self.alpha == other.alpha), Not(self.beta == other.beta))

    def __str__(self):
        """
        String representation is the qbit's identifier.
        :return: Qbit's identifier.
        """
        return f'({str(self.alpha)}, {str(self.beta)})'

    def get_identifier(self):
        """
        Get qbit's identifier.
        :return: Qbit's identifier.
        """
        return self.alpha.get_identifier().split('.')[0]

    def __repr__(self):
        """
        String representation of the qbit's values.
        :return: Qbit value string.
        """
        try:
            repr = f'({precision_format.format(self.alpha.r)} + {precision_format.format(self.alpha.i)}j)|0⟩' \
                   f' + ({precision_format.format(self.beta.r)} + {precision_format.format(self.beta.i)}j)|1⟩'
            if self.phi is not None and self.theta is not None:
                repr += f', φ = {precision_format.format(self.phi)}, θ = {precision_format.format(self.theta)}'
            return repr
        except TypeError:
            return f'({self.alpha.r} + {self.alpha.i}j)|0⟩' \
                   f' + ({self.beta.r} + {self.beta.i}j)|1⟩,' \
                   f' φ = {self.phi}, θ = {self.theta}'

    def to_complex_list(self):
        """
        Unfold a qbit into a 2-element list of complex values.
        :return: 2-element list of complex values.
        """
        return [self.alpha, self.beta]


def Qbit(identifier: str) -> QbitVal:
    """
    Generate a named qbit.
    :param identifier: chosen identifier.
    :return: Qbit.
    """
    return QbitVal(Complex('%s.alpha' % identifier),
                   Complex('%s.beta' % identifier),
                   Real('%s.phi' % identifier),
                   Real('%s.theta' % identifier))


def Qbits(identifiers: List[str]) -> List[QbitVal]:
    """
    Generate many named qbits.
    :param identifiers: chose identifiers.
    :return: List of qbits.
    """
    return [Qbit(identifier) for identifier in identifiers]
