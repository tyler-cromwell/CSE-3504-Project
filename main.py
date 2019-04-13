#!/usr/bin/env python3
import enum
import math

import numpy


A = numpy.matrix([
    [0.00, 0.70, 0.15, 0.15, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.70, 0.00, 0.30, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.60, 0.40, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.40, 0.60, 0.00, 0.00],
    [0.00, 0.00, 0.30, 0.00, 0.00, 0.00, 0.30, 0.10, 0.30, 0.00],
    [0.00, 0.50, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.50, 0.00],
    [0.00, 0.00, 0.00, 0.25, 0.00, 0.00, 0.00, 0.00, 0.00, 0.75],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.10, 0.00, 0.90],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00]
])


class Axis(enum.Enum):
    ROW = 0
    COLUMN = 1


def I(n=1):
    I = []
    index = 0

    for i in range(n):
        l = [0] * n
        l[index] = 1
        index += 1
        I.append(l)

    return numpy.matrix(I)


if __name__ == '__main__':
    print(type(A), A.shape)
    print(A)

    # Remove absorbing state A_24
    B = numpy.delete(
        numpy.delete(
            A,
            2,
            Axis.ROW.value
        ),
        4,
        Axis.COLUMN.value
    )
    print(type(B), B.shape)
    print(B)

    # Remove absorbing state B_88
    C = numpy.delete(
        numpy.delete(
            B,
            8,
            Axis.ROW.value
        ),
        8,
        Axis.COLUMN.value
    )
    print(type(C), C.shape)
    print(C)

    M = numpy.linalg.inv(I(8) - C)
    print(M)
