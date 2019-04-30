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


def compute_fundamental_matrix(matrix):
    B = numpy.delete(
        numpy.delete(
            matrix,
            9,
            Axis.ROW.value
        ),
        9,
        Axis.COLUMN.value
    )

    # Compute average number of visits
    return numpy.linalg.inv(I(int(math.sqrt(B.size))) - B)


def mean_execution_time(C, T):
    # Multiply time vector with the first row of the fundamental matrix
    U = numpy.delete(T, 9)
    R = (C[0, :] * U.T).A1

    # Compute total average completion time
    return T.A1[9] + R[0]


if __name__ == '__main__':
    numpy.set_printoptions(precision=5)

    ########################################
    # Question 1
    ########################################

    ########################################
    # Part B
    ########################################
    # Remove absorbing state A_99
    C = compute_fundamental_matrix(A)
    print('########################################')
    print('Part B')
    print('Average number of visits to each module:')
    print(C.A[0])
    print()


    ########################################
    # Part C
    ########################################
    total = mean_execution_time(C, T)
    print('########################################')
    print('Part C')
    print('Mean execution time:')
    print('{:.2f} sec'.format(total))
    print()


    ########################################
    # Part D
    ########################################
    U = numpy.delete(T, 9)
    l1 = C[0, :].A1
    l2 = U.T.A1

    print('########################################')
    print('Part D')
    print('Execution time per component:')
    for i in range(len(l1)):
        print('Component {:}: {:.2f} sec'.format(i+1, l1[i] * l2[i]))
    print('Component {:}: {:.2f} sec'.format(10, T.A1[9]))
    print()


    ########################################
    # Part E
    ########################################
    steps = [(i/20) for i in list(range(2, 19, 1))]

    print('########################################')
    print('Part E')

    print('P_5,7')
    E = A.copy()
    for p in steps:
        E.A[4][6] = p
        E.A[4][7] = q = 1 - p
        C = compute_fundamental_matrix(E)
        total = mean_execution_time(C, T)
        if p == A.A[4][6]:
            print('Result (P_5,7, P_5,8, mean): ({:.2f}, {:.2f}, {:.2f} sec) (original values)'.format(p, q, total))
        else:
            print('Result (P_5,7, P_5,8, mean): ({:.2f}, {:.2f}, {:.2f} sec)'.format(p, q, total))

    print('P_7,2')
    E = A.copy()
    for p in steps:
        E.A[6][1] = p
        E.A[6][8] = q = 1 - p
        C = compute_fundamental_matrix(E)
        total = mean_execution_time(C, T)
        if p == A.A[6][1]:
            print('Result (P_7,2, P_7,9, mean): ({:.2f}, {:.2f}, {:.2f} sec) (original values)'.format(p, q, total))
        else:
            print('Result (P_7,2, P_7,9, mean): ({:.2f}, {:.2f}, {:.2f} sec)'.format(p, q, total))

    print('P_8,4')
    E = A.copy()
    for p in steps:
        E.A[7][3] = p
        E.A[7][9] = q = 1 - p
        C = compute_fundamental_matrix(E)
        total = mean_execution_time(C, T)
        if p == A.A[7][3]:
            print('Result (P_8,4, P_8,10, mean): ({:.2f}, {:.2f}, {:.2f} sec) (original values)'.format(p, q, total))
        else:
            print('Result (P_8,4, P_8,10, mean): ({:.2f}, {:.2f}, {:.2f} sec)'.format(p, q, total))

    print('P_9,8')
    E = A.copy()
    for p in steps:
        E.A[8][7] = p
        E.A[8][9] = q = 1 - p
        C = compute_fundamental_matrix(E)
        total = mean_execution_time(C, T)
        if p == A.A[8][7]:
            print('Result (P_9,8, P_9,10, mean): ({:.2f}, {:.2f}, {:.2f} sec) (original values)'.format(p, q, total))
        else:
            print('Result (P_9,8, P_9,10, mean): ({:.2f}, {:.2f}, {:.2f} sec)'.format(p, q, total))


    ########################################
    # Part F
    ########################################
    print()
    print('########################################')
    print('Part F')
    print()

    X = T.copy()    # -10%
    Y = T.copy()    # +10%
    index = -1
    widest = 0

    print('Default:', T, '{:.2f} sec'.format(mean_execution_time(C, T)))
    print()

    for i in range(len(X.A1)):
        original = X.A1[i]
        original = Y.A1[i]

        X.A1[i] = X.A1[i] * 0.9
        Y.A1[i] = Y.A1[i] * 1.1

        lower = mean_execution_time(C, X)
        upper = mean_execution_time(C, Y)
        diff = upper - lower

        if diff >= widest:
            widest = diff
            index = i

        print(X, '{:.2f} sec'.format(lower))
        print(Y, '{:.2f} sec'.format(upper))
        print('Module {:}, Difference {:.2f} sec'.format(i, diff))
        print()

        X.A1[i] = original
        Y.A1[i] = original

    print('Most sensitive: Module {:} with a {:.2f} sec difference'.format(index, widest))
