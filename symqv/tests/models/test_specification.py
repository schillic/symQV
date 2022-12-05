from symqv.lib.models.specification import Specification


def test_from_file():
    path = '../../../benchmarks/symqv/specs/teleportv3.qspec'
    spec = Specification.read_from_file(path)

    print(spec)

    path = '../../../benchmarks/symqv/specs/qft.qspec'
    spec = Specification.read_from_file(path)

    print(spec)


if __name__ == '__main__':
    test_from_file()
