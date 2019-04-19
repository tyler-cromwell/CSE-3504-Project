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
T = numpy.matrix([
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


def iv(n=1):
    return numpy.matrix([1] * n)


if __name__ == '__main__':
    numpy.set_printoptions(precision=5)

    ########################################
    # Part B
    ########################################
    # Remove absorbing state A_99
    B = numpy.delete(
        numpy.delete(
            A,
            9,
            Axis.ROW.value
        ),
        9,
        Axis.COLUMN.value
    )

    # Compute average number of visits
    C = numpy.linalg.inv(I(int(math.sqrt(B.size))) - B)
    print('Average number of visits to each module:')
    print(C)
    print()


    ########################################
    # Part C
    ########################################
    # Multiply time vector with the first row of the fundamental matrix
    U = numpy.delete(T, 9)
    R = (C[0, :] * U.T).A1

    # Compute total average completion time
    total = T.A1[9] + R[0]

    print('Average completion time:')
    print('{:.2f} seconds'.format(total))
    print()


    ########################################
    # Part D
    ########################################
    l1 = C[0, :].A1
    l2 = U.T.A1

    print('Execution time per component:')
    for i in range(len(l1)):
        print('Component {:}: {:.2f} seconds'.format(i+1, l1[i] * l2[i]))
    print('Component {:}: {:.2f} seconds'.format(10, T.A1[9]))
    print()
