#!/usr/bin/env python3
import enum
import math

import numpy


# Transition probabilities from i to j
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


# Time (seconds) per visit
T = numpy.array([
    4.0,
    5.0,
    2.0,
    6.0,
    1.0,
    2.0,
    3.0,
    4.0,
    2.0,
    8.0
])


class Axis(enum.Enum):
    ROW = 0
    COLUMN = 1


# Generate Identity Matrix of size nxn
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
    ########################################
    # Part B
    # TODO: Account for Absorbing States!
    ########################################
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

    # Compute average number of visits
    M = numpy.linalg.inv(I(8) - C)
    print('Average number of visits to each module:')
    print(M)
    print()


    ########################################
    # Part C
    ########################################
    U = numpy.delete(T, 2)
    U = numpy.delete(U, 8)
    R = M.dot(U).A1

    total = T[2] + T[9]
    for r in R: total += r

    print('Average completion time:')
    print('{:.1f}'.format(total), 'seconds (approximately)')
    print()
