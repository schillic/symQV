from quavl.lib.constants import X_matrix


def test_tensordot():
    import numpy as np
    from scipy.linalg import norm

    H_matrix = 1 / np.sqrt(2) * np.array([[1, 1],
                                          [1, -1]])

    CNOT_matrix = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1],
                            [0, 0, 1, 0]])

    CNOT_tensor = np.reshape(CNOT_matrix, (2, 2, 2, 2))

    class Reg:
        def __init__(self, n):
            self.n = n
            self.psi = np.zeros((2,) * n)
            self.psi[(0,) * n] = 1

    def X(i, reg):
        reg.psi = np.tensordot(X_matrix, reg.psi, (1, i))
        reg.psi = np.moveaxis(reg.psi, 0, i)

    def H(i, reg):
        reg.psi = np.tensordot(H_matrix, reg.psi, (1, i))
        reg.psi = np.moveaxis(reg.psi, 0, i)

    def CNOT(control, target, reg):
        reg.psi = np.tensordot(CNOT_tensor, reg.psi, ((2, 3), (control, target)))
        reg.psi = np.moveaxis(reg.psi, (0, 1), (control, target))

    def measure(i, reg):
        projectors = [np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]])]

        def project(i, j, reg):
            projected = np.tensordot(projectors[j], reg.psi, (1, i))
            return np.moveaxis(projected, 0, i)

        projected = project(i, 0, reg)
        norm_projected = norm(projected.flatten())
        if np.random.random() < norm_projected ** 2:
            reg.psi = projected / norm_projected
            return 0
        else:
            projected = project(i, 1, reg)
            reg.psi = projected / norm(projected)
            return 1

    # Example of final usage: create uniform superposition
    reg = Reg(4)
    print(reg.psi.flatten())

    X(3, reg)
    print(reg.psi.flatten())

    for i in range(reg.n):
        H(i, reg)

    print(reg.psi.flatten())


if __name__ == "__main__":
    test_tensordot()
