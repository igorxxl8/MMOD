from lab1 import rand_basic_value_gen
from lab2.discrete_system_random_values_func import DIST_TABLE, A, B


def imitate(dist):
    r = rand_basic_value_gen.next_value()
    left_bound = 0
    right_bound = dist[0]
    xi = 0
    while not left_bound <= r < right_bound:
        left_bound = right_bound
        xi += 1
        right_bound += dist[xi]
    return xi


def two_dim_discrete_gen(N):
    x = []
    y = []
    x_dist = [sum(x) for x in DIST_TABLE]
    for i in range(N):
        xi = imitate(x_dist)
        _x = A[xi]

        xp = sum(DIST_TABLE[xi])
        y_dist = [_y / xp for _y in DIST_TABLE[xi]]
        _y = B[imitate(y_dist)]
        x.append(_x)
        y.append(_y)
    return x, y, build_empiric_distribution_matrix(x, y, N), DIST_TABLE


def build_empiric_distribution_matrix(x, y, N):
    n = len(DIST_TABLE)
    m = len(DIST_TABLE[0])
    matrix = []
    for i in range(n):
        row = [0] * m
        matrix.append(row)

    for i, xi in enumerate(x):
        yi = y[i]
        matrix[A.index(xi)][B.index(yi)] += 1

    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            matrix[i][j] = matrix[i][j] / N

    return matrix
