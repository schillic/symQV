from z3 import Int


class KetVal:
    def __init__(self, b, n):
        """
        Constructor for a Ket.
        :param b: binary value.
        :param n: arity.
        """
        self.b = b
        self.n = n

    def __mul__(self, other):
        """
        Kronecker product.
        :param other: other Ket.
        :return: Kronecker Product.
        """
        return KetVal(self.b * 2 ** (self.n + 1) + other.b, self.n + other.n)


def Ket(identifier: str) -> KetVal:
    """
    Generate a named Ket.
    :param identifier: chosen identifier.
    :return: Ket.
    """
    return KetVal(Int(f'{identifier}.b'), Int(f'{identifier}.n'))