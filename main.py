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


# Generate an identity vector of size 1xn
def iv(n=1):
    return numpy.matrix([1] * n)


# Specifically for Question 1
def compute_fundamental_matrix(A):
    Q = numpy.delete(
        numpy.delete(
            A,
            9,
            Axis.ROW.value
        ),
        9,
        Axis.COLUMN.value
    )

    # Compute average number of visits
    return numpy.linalg.inv(I(int(math.sqrt(Q.size))) - Q)


def compute_mean_execution_time(C, T):
    # Multiply time vector with the first row of the fundamental matrix
    U = numpy.delete(T, 9)
    R = (C[0, :] * U.T).A1

    # Compute total average completion time
    return T.A1[9] + R[0]


def compute_steady_state(B, cols):
    # Apply normalization condition
    return numpy.full((1, cols), 1 / cols) * B


def compute_reliability(S):
    Z = numpy.matrix([(0.85 + 0.01*(i+1)) for i in range(10)])
    return (Z * S.T).A1.tolist()[0]


if __name__ == '__main__':
    numpy.set_printoptions(precision=5)

    print('########################################')
    print('########################################')
    print('Question 1')
    print('########################################')
    print('########################################')
    print()

    print('########################################')
    print('Part B')
    # Remove absorbing state A_99
    M = compute_fundamental_matrix(A)
    print('Average number of visits to each module:')
    print(M.A[0])
    print()

    print('########################################')
    print('Part C')
    total = compute_mean_execution_time(M, T)
    print('Mean execution time: {:.2f} sec'.format(total))
    print()

    print('########################################')
    print('Part D')
    U = numpy.delete(T, 9)
    l1 = M[0, :].A1
    l2 = U.T.A1

    print('Execution time per component:')
    for i in range(len(l1)):
        print('Component {:}: {:.2f} sec'.format(i+1, l1[i] * l2[i]))
    print('Component {:}: {:.2f} sec'.format(10, T.A1[9]))
    print()

    print('########################################')
    print('Part E')
    steps = [(i/20) for i in list(range(2, 19, 1))]
    n = len(steps)

    print('P_5,7')
    E = A.copy()
    first1 = 0
    last1 = 0

    for i in range(len(steps)):
        p = steps[i]
        E.A[4][6] = p
        E.A[4][7] = q = 1 - p

        M = compute_fundamental_matrix(E)
        total = compute_mean_execution_time(M, T)

        if p == A.A[4][6]:
            print('Result (P_5,7, P_5,8, mean): ({:.2f}, {:.2f}, {:.2f} sec) (original values)'.format(p, q, total))
        else:
            print('Result (P_5,7, P_5,8, mean): ({:.2f}, {:.2f}, {:.2f} sec)'.format(p, q, total))

        if i == 0:
            first1 = total
        if i == n-1:
            last1 = total

    print('P_7,2')
    E = A.copy()
    first2 = 0
    last2 = 0

    for i in range(len(steps)):
        p = steps[i]
        E.A[6][1] = p
        E.A[6][8] = q = 1 - p

        M = compute_fundamental_matrix(E)
        total = compute_mean_execution_time(M, T)

        if p == A.A[6][1]:
            print('Result (P_7,2, P_7,9, mean): ({:.2f}, {:.2f}, {:.2f} sec) (original values)'.format(p, q, total))
        else:
            print('Result (P_7,2, P_7,9, mean): ({:.2f}, {:.2f}, {:.2f} sec)'.format(p, q, total))

        if i == 0:
            first2 = total
        if i == n-1:
            last2 = total

    print('P_8,4')
    E = A.copy()
    first3 = 0
    last3 = 0

    for i in range(len(steps)):
        p = steps[i]
        E.A[7][3] = p
        E.A[7][9] = q = 1 - p

        M = compute_fundamental_matrix(E)
        total = compute_mean_execution_time(M, T)

        if p == A.A[7][3]:
            print('Result (P_8,4, P_8,10, mean): ({:.2f}, {:.2f}, {:.2f} sec) (original values)'.format(p, q, total))
        else:
            print('Result (P_8,4, P_8,10, mean): ({:.2f}, {:.2f}, {:.2f} sec)'.format(p, q, total))

        if i == 0:
            first3 = total
        if i == n-1:
            last3 = total

    print('P_9,8')
    E = A.copy()
    first4 = 0
    last4 = 0

    for i in range(len(steps)):
        p = steps[i]
        E.A[8][7] = p
        E.A[8][9] = q = 1 - p

        M = compute_fundamental_matrix(E)
        total = compute_mean_execution_time(M, T)

        if p == A.A[8][7]:
            print('Result (P_9,8, P_9,10, mean): ({:.2f}, {:.2f}, {:.2f} sec) (original values)'.format(p, q, total))
        else:
            print('Result (P_9,8, P_9,10, mean): ({:.2f}, {:.2f}, {:.2f} sec)'.format(p, q, total))

        if i == 0:
            first4 = total
        if i == n-1:
            last4 = total

    D = {
        abs(first1 - last1) : 'P_5,7',
        abs(first2 - last2) : 'P_7,2',
        abs(first3 - last3) : 'P_8,4',
        abs(first4 - last4) : 'P_9,8'
    }

    diff = max(
        abs(first1 - last1),
        abs(first2 - last2),
        abs(first3 - last3),
        abs(first4 - last4)
    )

    print()
    print('Transition {:} has the greatest impact with difference {:.4f}'.format(D[diff], diff))
    print()

    print('########################################')
    print('Part F')
    print()

    X = T.copy()    # -10%
    Y = T.copy()    # +10%
    index = -1
    widest = 0

    print('Default:', T, '{:.2f} sec'.format(compute_mean_execution_time(M, T)))
    print()

    for i in range(len(X.A1)):
        original = X.A1[i]
        original = Y.A1[i]

        X.A1[i] = X.A1[i] * 0.9
        Y.A1[i] = Y.A1[i] * 1.1

        lower = compute_mean_execution_time(M, X)
        upper = compute_mean_execution_time(M, Y)
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
    print()


    print('########################################')
    print('########################################')
    print('Question 2')
    print('########################################')
    print('########################################')
    print()

    B = A.copy()
    B.A[9][1] = 0.5
    B.A[9][2] = 0.5
    B.A[9][9] = 0

    print('########################################')
    print('Part A')
    S = compute_steady_state(B, 10)
    print('Steady State:', numpy.around(S.A1, 4).tolist())
    print()

    print('########################################')
    print('Part B')
    r = compute_reliability(S)
    print('Reliability: {:.5f}'.format(r))
    print()

    print('########################################')
    print('Part C')
    print()
    n = len(steps)

    print('P_5,7')
    Q = B.copy()
    first1 = 0
    last1 = 0

    for i in range(n):
        p = steps[i]
        Q.A[4][6] = p
        Q.A[4][7] = q = 1 - p

        S = compute_steady_state(Q, 10)
        r = compute_reliability(S)

        if p == B.A[4][6]:
            print('Result (P_5,7, P_5,8, reliability): ({:.2f}, {:.2f}, {:.4f}) (original values)'.format(p, q, r))
        else:
            print('Result (P_5,7, P_5,8, reliability): ({:.2f}, {:.2f}, {:.4f})'.format(p, q, r))

        if i == 0:
            first1 = r
        if i == n-1:
            last1 = r

    print('P_7,2')
    Q = B.copy()
    first2 = 0
    last2 = 0

    for i in range(n):
        p = steps[i]
        Q.A[6][1] = p
        Q.A[6][8] = q = 1 - p

        S = compute_steady_state(Q, 10)
        r = compute_reliability(S)

        if p == B.A[6][1]:
            print('Result (P_7,2, P_7,9, reliability): ({:.2f}, {:.2f}, {:.4f}) (original values)'.format(p, q, r))
        else:
            print('Result (P_7,2, P_7,9, reliability): ({:.2f}, {:.2f}, {:.4f})'.format(p, q, r))

        if i == 0:
            first2 = r
        if i == n-1:
            last2 = r

    print('P_8,4')
    Q = B.copy()
    first3 = 0
    last3 = 0

    for i in range(n):
        p = steps[i]
        Q.A[7][3] = p
        Q.A[7][9] = q = 1 - p

        S = compute_steady_state(Q, 10)
        r = compute_reliability(S)

        if p == B.A[7][3]:
            print('Result (P_8,4, P_8,10, reliability): ({:.2f}, {:.2f}, {:.4f}) (original values)'.format(p, q, r))
        else:
            print('Result (P_8,4, P_8,10, reliability): ({:.2f}, {:.2f}, {:.4f})'.format(p, q, r))

        if i == 0:
            first3 = r
        if i == n-1:
            last3 = r

    print('P_9,8')
    Q = B.copy()
    first4 = 0
    last4 = 0

    for i in range(n):
        p = steps[i]
        Q.A[8][7] = p
        Q.A[8][9] = q = 1 - p

        S = compute_steady_state(Q, 10)
        r = compute_reliability(S)

        if p == B.A[8][7]:
            print('Result (P_9,8, P_9,10, reliability): ({:.2f}, {:.2f}, {:.4f}) (original values)'.format(p, q, r))
        else:
            print('Result (P_9,8, P_9,10, reliability): ({:.2f}, {:.2f}, {:.4f})'.format(p, q, r))

        if i == 0:
            first4 = r
        if i == n-1:
            last4 = r

    D = {
        abs(first1 - last1) : 'P_5,7',
        abs(first2 - last2) : 'P_7,2',
        abs(first3 - last3) : 'P_8,4',
        abs(first4 - last4) : 'P_9,8'
    }

    diff = max(
        abs(first1 - last1),
        abs(first2 - last2),
        abs(first3 - last3),
        abs(first4 - last4)
    )

    print()
    print('Transition {:} has the greatest impact with difference {:.4f}'.format(D[diff], diff))
