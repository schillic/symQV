from typing import List

from z3 import is_rational_value, simplify, Real, Not, RealVal, And, ArithRef
from quavl.lib.globals import precision_format


class ComplexVal:
    def __init__(self, r: float, i: float = 0):
        """
        Constructor for a complex number.
        :param r: real part.
        :param i: imaginary part (optional).
        """
        self.r = r
        self.i = i

    def __add__(self, other: 'ComplexVal'):
        """
        Addition.
        :param other: other complex number.
        :return: Sum.
        """
        other = _to_complex(other)
        return ComplexVal(self.r + other.r, self.i + other.i)

    def __mod__(self, other: int):
        """
        Modulus.
        :param other: integer.
        :return: Modulus.
        """
        raise Exception('Not implemented.')

    def __radd__(self, other):
        """
        Addition with a real number.
        :param other: other real number
        :return: Sum.
        """
        other = _to_complex(other)
        return ComplexVal(other.r + self.r, other.i + self.i)

    def __sub__(self, other: 'ComplexVal'):
        """
        Subtraction.
        :param other: other complex number.
        :return: Difference.
        """
        other = _to_complex(other)
        return ComplexVal(self.r - other.r, self.i - other.i)

    def __rsub__(self, other):
        """
        Subtraction with a real number.
        :param other: other real number.
        :return: Difference.
        """
        other = _to_complex(other)
        return ComplexVal(other.r - self.r, other.i - self.i)

    def __mul__(self, other):
        """
        Multiplication.
        :param other: other complex number.
        :return: Product.
        """
        other = _to_complex(other)
        return ComplexVal(self.r * other.r - self.i * other.i, self.r * other.i + self.i * other.r)

    def inv(self):
        """
        Complex inverse.
        :return: Complex inverse.
        """
        den = self.r * self.r + self.i * self.i
        return ComplexVal(self.r / den, -self.i / den)

    def __div__(self, other):
        """
        Division with a complex number.
        :param other: other complex number.
        :return: Fraction.
        """
        inv_other = _to_complex(other).inv()
        return self.__mul__(inv_other)

    def __rdiv__(self, other):
        """
        Division with a real number.
        :param other: other real number.
        :return: Fraction.
        """
        other = _to_complex(other)
        return self.inv().__mul__(other)

    def __eq__(self, other: 'ComplexVal'):
        """
        Equality with a complex number.
        :param other: other complex number.
        :return: Equality.
        """
        other = _to_complex(other)
        return And(self.r == other.r, self.i == other.i)

    def __neq__(self, other: 'ComplexVal'):
        """
        Unequality with a complex number.
        :param other: other complex number.
        :return: Unequality.
        """
        return Not(self.__eq__(other))

    def simplify(self):
        """
        Simplify a complex number.
        :return: simplification.
        """
        return ComplexVal(simplify(self.r), simplify(self.i))

    def repr_i(self):
        """
        :return: Representation of imaginary part.
        """
        return f'{precision_format.format(self.i)}*I'

    def __str__(self):
        """
        :return: Identifier.
        """
        return f'{str(self.r)} + {str(self.i)}*I'

    def get_identifier(self):
        """
        :return: Identifier.
        """
        return str(self.r).split('.')[0]

    def __repr__(self):
        """
        :return: Representation of complex number.
        """
        if isinstance(self.r, ArithRef):
            return str(self)
        elif _is_zero(self.i):
            return str(self.r)
        elif _is_zero(self.r):
            return self.repr_i()
        else:
            return f'{precision_format.format(self.r)} + {self.repr_i()}'


def Complex(identifier: str) -> ComplexVal:
    """
    Generate a named complex number.
    :param identifier: chosen identifier.
    :return: Complex number.
    """
    return ComplexVal(Real(f'{identifier}.r'), Real(f'{identifier}.i'))


def Complexes(identifiers: List[str]) -> List[ComplexVal]:
    """
    Generate named complex numbers.
    :param identifiers: chosen identifiers.
    :return: List of complex numbers.
    """
    return [Complex(identifier) for identifier in identifiers]


def evaluate_cexpr(m, e):
    return ComplexVal(m[e.r], m[e.i])


def _to_complex(a):
    """
    :param a: real or complex value.
    :return: Complex value.
    """
    if isinstance(a, ComplexVal):
        return a
    elif isinstance(a, complex):
        return ComplexVal(a.real, a.imag)
    else:
        return ComplexVal(a, RealVal(0))


def _is_zero(a):
    """
    Check if a number is zero.
    :param a: integer or rational number.
    :return: True if zero, false otherwise.
    """
    return (isinstance(a, int) and a == 0) or (is_rational_value(a) and a.numerator_as_long() == 0)
