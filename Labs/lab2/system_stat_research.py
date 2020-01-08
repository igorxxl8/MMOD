from math import sqrt, log10
from scipy.stats import norm, chi2
from matplotlib import pyplot as plt
from lab2 import continuous_system_func


def histogram(values, title, bins=10):
    plt.hist(values, density=True, bins=bins)
    plt.title(title)
    plt.show()


def X_Y(system):
    X = []
    Y = []

    for i in range(len(system)):
        X.append(system[i][0])
        Y.append(system[i][1])

    return X, Y


def points_estimate_continuous(X, Y):
    n = len(X)

    mx = continuous_system_func.x_m()
    my = continuous_system_func.y_m()
    dx = continuous_system_func.x_d()
    dy = continuous_system_func.y_d()

    _mx = me(X, n)
    _my = me(Y, n)
    _dx = disp(X, _mx, n)
    _dy = disp(Y, _my, n)

    print('Point estimates:')
    print('M[X] = {} : M[X]* = {}'.format(mx, _mx))
    print('M[Y] = {} : M[Y]* = {}'.format(my, _my))
    print('D[X] = {} : D[X]* = {}'.format(dx, _dx))
    print('D[Y] = {} : D[Y]* = {}'.format(dy, _dy))


def me(values, n):
    return sum(values) / n


def disp(values, m, n):
    return sum([(value - m) ** 2 for value in values]) / (n - 1)


def intervals_estimate_continuous(X, Y, q):
    n = len(X)

    _mx = me(X, n)
    _my = me(Y, n)
    _dx = disp(X, _mx, n)
    _dy = disp(Y, _my, n)
    _sx = sqrt(_dx)
    _sy = sqrt(_dy)

    kx = norm.ppf(q) * _sx / sqrt(n)
    ky = norm.ppf(q) * _sy / sqrt(n)

    print('\nConfidence interval for ME:'.format(q))
    print('X: {} < M < {}'.format(_mx - kx, _mx + kx))
    print('Y: {} < M < {}'.format(_my - ky, _my + ky))

    print('\nConfidence interval for Dispersion:'.format(q))
    excess_x, excess_y = excess(X, Y, n)
    kx = norm.ppf(q) * sqrt((excess_x + 2) / n) * _dx
    ky = norm.ppf(q) * sqrt((excess_y + 2) / n) * _dy

    print('X: {} < D < {}'.format(_dx - kx, _dx + kx))
    print('Y: {} < M < {}'.format(_dy - ky, _dy + ky))


def excess(X, Y, n):
    mx = continuous_system_func.x_m()
    my = continuous_system_func.y_m()
    dx = continuous_system_func.x_d()
    dy = continuous_system_func.y_d()

    mu_x = sum([(value - mx) ** 4 for value in X]) / n
    mu_y = sum([(value - my) ** 4 for value in Y]) / n

    return mu_x / dx ** 2 - 3, mu_y / dy ** 2 - 3


def check_correlation_continues(x_values, y_values):
    n = len(x_values)

    _mx = me(x_values, n)
    _my = me(y_values, n)
    _dx = disp(x_values, _mx, n)
    _dy = disp(y_values, _my, n)

    covariance = sum([(x_values[i] - _mx) * (y_values[i] - _my) / n ** 2 for i in range(n)])

    correlation = covariance / sqrt(_dx * _dy)
    print('\nCorrelation: {}'.format(correlation))


def z_test_continues(X, Y):
    n = len(X)

    _mx = me(X, n)
    _my = me(Y, n)
    mx = continuous_system_func.x_m()
    my = continuous_system_func.y_m()
    dx = continuous_system_func.x_d()
    dy = continuous_system_func.y_d()

    zx = (_mx - mx) / sqrt(dx * n)
    zy = (_my - my) / sqrt(dy * n)

    print('\nZ-test:')
    print('X: {}'.format(zx))
    print('Y: {}'.format(zy))


def f_test_continues(X, Y):
    n = len(X)

    _mx = me(X, n)
    _my = me(Y, n)
    _dx = disp(X, _mx, n)
    _dy = disp(Y, _my, n)
    dx = continuous_system_func.x_d()
    dy = continuous_system_func.y_d()

    if _dx > dx:
        f_x = _dx / dx
    else:
        f_x = dx / _dx

    if _dy > dy:
        f_y = _dy / dy
    else:
        f_y = dy / _dy

    print('\nF-test:')
    print('X: {}'.format(f_x))
    print('Y: {}'.format(f_y))


def point_estimate_discrete(dist_table, x_y_matrix):
    mx = me_disc(dist_table, True)
    my = me_disc(dist_table, False)
    dx = disp_disc(dist_table, mx, True)
    dy = disp_disc(dist_table, my, False)

    _mx = me_disc(x_y_matrix, True)
    _my = me_disc(x_y_matrix, False)
    _dx = disp_disc(x_y_matrix, _mx, True)
    _dy = disp_disc(x_y_matrix, _my, False)

    print('Point estimates:')
    print('M[X] = {} : M[X]* = {}'.format(mx, _mx))
    print('M[Y] = {} : M[Y]* = {}'.format(my, _my))
    print('D[X] = {} : D[X]* = {}'.format(dx, _dx))
    print('D[Y] = {} : D[Y]* = {}'.format(dy, _dy))


def me_disc(matrix, is_for_x):
    m = 0

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if is_for_x:
                m += matrix[i][j] * i
            else:
                m += matrix[i][j] * j

    return m


def disp_disc(matrix, m, is_for_x):
    d = 0

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if is_for_x:
                d += matrix[i][j] * (i - m) ** 2
            else:
                d += matrix[i][j] * (j - m) ** 2

    return d


def intervals_estimate_discrete(dist_table, x_y_matrix, n, q):
    _mx = me_disc(x_y_matrix, True)
    _my = me_disc(x_y_matrix, False)
    _dx = disp_disc(x_y_matrix, _mx, True)
    _dy = disp_disc(x_y_matrix, _my, False)
    _sx = sqrt(_dx)
    _sy = sqrt(_dy)

    kx = norm.ppf(q) * _sx / sqrt(n)
    ky = norm.ppf(q) * _sy / sqrt(n)

    print('\nConfidence interval for ME:'.format(q))
    print('X: {} < M < {}'.format(_mx - kx, _mx + kx))
    print('Y: {} < M < {}'.format(_my - ky, _my + ky))

    print('\nConfidence interval for Dispersion:'.format(q))
    excess_x, excess_y = excess_disc(dist_table, x_y_matrix, n)
    kx = norm.ppf(q) * sqrt((excess_x + 2) / n) * _dx
    ky = norm.ppf(q) * sqrt((excess_y + 3) / n) * _dy

    print('X: {} < D < {}'.format(_dx - kx, _dx + kx))
    print('Y: {} < D < {}'.format(_dy - ky, _dy + ky))


def excess_disc(dist_table, x_y_matrix, n):
    _n = len(dist_table)
    _m = len(dist_table[0])

    mx = me_disc(dist_table, True)
    my = me_disc(dist_table, False)
    dx = disp_disc(dist_table, mx, True)
    dy = disp_disc(dist_table, my, False)

    x_density = []
    for i in range(_m):
        density = 0

        for j in range(_n):
            density += x_y_matrix[j][i]
        x_density.append(density * n)

    y_density = [sum(row) * n for row in x_y_matrix]

    mu_x = sum([(value - mx) ** 4 * x_density[value] for value in range(_m)]) / n
    mu_y = sum([(value - my) ** 4 * y_density[value] for value in range(_n)]) / n

    return mu_x / dx ** 2 - 3, mu_y / dy ** 2 - 3


def check_correlation_discrete(x_y_matrix):
    _mx = me_disc(x_y_matrix, True)
    _my = me_disc(x_y_matrix, False)
    _dx = disp_disc(x_y_matrix, _mx, True)
    _dy = disp_disc(x_y_matrix, _my, False)

    covariance = 0
    for i in range(len(x_y_matrix)):
        for j in range(len(x_y_matrix[0])):
            covariance += (i - _mx) * (i - _my) * x_y_matrix[i][j]

    correlation = covariance / sqrt(_dx * _dy)
    print('\nCorrelation: {}'.format(correlation))


def z_test_discrete(dist_table, x_y_matrix, n):
    _mx = me_disc(x_y_matrix, True)
    _my = me_disc(x_y_matrix, False)
    mx = me_disc(dist_table, True)
    my = me_disc(dist_table, False)
    dx = disp_disc(dist_table, mx, True)
    dy = disp_disc(dist_table, my, False)

    zx = (_mx - mx) / sqrt(dx * n)
    zy = (_my - my) / sqrt(dy * n)

    print('\nZ-test:')
    print('X: {}'.format(zx))
    print('Y: {}'.format(zy))


def f_test_discrete(dist_table, x_y_matrix):
    _mx = me_disc(x_y_matrix, True)
    _my = me_disc(x_y_matrix, False)
    _dx = disp_disc(x_y_matrix, _mx, True)
    _dy = disp_disc(x_y_matrix, _my, False)
    mx = me_disc(dist_table, True)
    my = me_disc(dist_table, False)
    dx = disp_disc(dist_table, mx, True)
    dy = disp_disc(dist_table, my, False)

    if _dx > dx:
        f_x = _dx / dx
    else:
        f_x = dx / _dx

    if _dy > dy:
        f_y = _dy / dy
    else:
        f_y = dy / _dy

    print('\nF-test:')
    print('X: {}'.format(f_x))
    print('Y: {}'.format(f_y))


def continuous_research(x_y_system):
    X, Y = X_Y(x_y_system)
    n = len(X)
    k = int(sqrt(n)) if n <= 100 else int(4 * log10(n))

    histogram(X, 'X', k)
    histogram(Y, 'Y', k)

    points_estimate_continuous(X, Y)
    intervals_estimate_continuous(X, Y, 0.95)
    check_correlation_continues(X, Y)

    z_test_continues(X, Y)
    f_test_continues(X, Y)


def _print(matrix, title=None):
    if title:
        print(f"{title}:")

    for row in matrix:
        print(*row, sep="\t", end="\n")


def check_discrete_distribution_hypothesis(t, edm, N, alpha=0.05):
    n = len(t)
    M = n ** 2

    chi2_val = 0
    for i in range(n):
        for j in range(n):
            chi2_val += ((t[i][j] - edm[i][j]) ** 2) / t[i][j]

    chi2_val = N * chi2_val
    print("chi^2 = ", chi2_val)

    chi2_table_val = chi2.isf(alpha,  M - 1)

    h0 = chi2_val < chi2_table_val
    print(f"{chi2_val} < {chi2_table_val} = ", h0)
    if h0:
        print("Hypothesis H0 accepted")
    else:
        print("Hypothesis H0 refused")


def discrete_research(x, y, edm, dist_table, n):
    _print(edm, "Empiric Distribution Matrix")
    histogram(x, "X")
    histogram(y, "Y")
    point_estimate_discrete(dist_table, edm)
    intervals_estimate_discrete(dist_table, edm, n, 0.95)
    check_correlation_discrete(edm)

    z_test_discrete(dist_table, edm, n)
    f_test_discrete(dist_table, edm)
