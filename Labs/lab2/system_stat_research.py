from math import sqrt, log10
from scipy.stats import norm
from matplotlib import pyplot as plt
from lab2 import continuous_system_func
from lab2.discrete_system_random_values_func import probability_matrix


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

    m_x = continuous_system_func.x_m()
    m_y = continuous_system_func.y_m()
    d_x = continuous_system_func.x_d()
    d_y = continuous_system_func.y_d()

    _m_x = me(X, n)
    _m_y = me(Y, n)
    _d_x = disp(X, _m_x, n)
    _d_y = disp(Y, _m_y, n)

    print('Point estimates:')
    print('M[X] = {} : M[X]* = {}'.format(m_x, _m_x))
    print('M[Y] = {} : M[Y]* = {}'.format(m_y, _m_y))
    print('D[X] = {} : D[X]* = {}'.format(d_x, _d_x))
    print('D[Y] = {} : D[Y]* = {}'.format(d_y, _d_y))


def me(values, n):
    return sum(values) / n


def disp(values, m, n):
    return sum([(value - m) ** 2 for value in values]) / (n - 1)


def intervals_estimate_continuous(X, Y, q):
    n = len(X)

    _m_x = me(X, n)
    _m_y = me(Y, n)
    _d_x = disp(X, _m_x, n)
    _d_y = disp(Y, _m_y, n)
    _s_x = sqrt(_d_x)
    _s_y = sqrt(_d_y)

    k_x = norm.ppf(q) * _s_x / sqrt(n)
    k_y = norm.ppf(q) * _s_y / sqrt(n)

    print('\nConfidence interval for ME (quantile - {}):'.format(q))
    print('X: {} < M < {}'.format(_m_x - k_x, _m_x + k_x))
    print('Y: {} < M < {}'.format(_m_y - k_y, _m_y + k_y))

    print('\nConfidence interval for Dispersion (quantile - {}):'.format(q))
    excess_x, excess_y = excess(X, Y, n)
    k_x = norm.ppf(q) * sqrt((excess_x + 2) / n) * _d_x
    k_y = norm.ppf(q) * sqrt((excess_y + 2) / n) * _d_y

    print('X: {} < D < {}'.format(_d_x - k_x, _d_x + k_x))
    print('Y: {} < M < {}'.format(_d_y - k_y, _d_y + k_y))


def excess(X, Y, n):
    m_x = continuous_system_func.x_m()
    m_y = continuous_system_func.y_m()
    d_x = continuous_system_func.x_d()
    d_y = continuous_system_func.y_d()

    mu_x = sum([(value - m_x) ** 4 for value in X]) / n
    mu_y = sum([(value - m_y) ** 4 for value in Y]) / n

    return mu_x / d_x ** 2 - 3, mu_y / d_y ** 2 - 3


def check_correlation_continues(x_values, y_values):
    n = len(x_values)

    _m_x = me(x_values, n)
    _m_y = me(y_values, n)
    _d_x = disp(x_values, _m_x, n)
    _d_y = disp(y_values, _m_y, n)

    covariance = sum([(x_values[i] - _m_x) * (y_values[i] - _m_y) / n ** 2 for i in range(n)])

    correlation = covariance / sqrt(_d_x * _d_y)
    print('\nCorrelation: {}'.format(correlation))


def z_test_continues(X, Y):
    n = len(X)

    _m_x = me(X, n)
    _m_y = me(Y, n)
    m_x = continuous_system_func.x_m()
    m_y = continuous_system_func.y_m()
    d_x = continuous_system_func.x_d()
    d_y = continuous_system_func.y_d()

    z_x = (_m_x - m_x) / sqrt(d_x * n)
    z_y = (_m_y - m_y) / sqrt(d_y * n)

    print('\nZ-test:')
    print('X: {}'.format(z_x))
    print('Y: {}'.format(z_y))


def f_test_continues(X, Y):
    n = len(X)

    _m_x = me(X, n)
    _m_y = me(Y, n)
    _d_x = disp(X, _m_x, n)
    _d_y = disp(Y, _m_y, n)
    d_x = continuous_system_func.x_d()
    d_y = continuous_system_func.y_d()

    if _d_x > d_x:
        f_x = _d_x / d_x
    else:
        f_x = d_x / _d_x

    if _d_y > d_y:
        f_y = _d_y / d_y
    else:
        f_y = d_y / _d_y

    print('\nF-test:')
    print('X: {}'.format(f_x))
    print('Y: {}'.format(f_y))


def point_estimate_discrete(distribution_table, x_y_matrix):
    m_x = me_disc(distribution_table, True)
    m_y = me_disc(distribution_table, False)
    d_x = disp_disc(distribution_table, m_x, True)
    d_y = disp_disc(distribution_table, m_y, False)

    _m_x = me_disc(x_y_matrix, True)
    _m_y = me_disc(x_y_matrix, False)
    _d_x = disp_disc(x_y_matrix, _m_x, True)
    _d_y = disp_disc(x_y_matrix, _m_y, False)

    print('Point estimates:')
    print('M[X] = {} : M[X]* = {}'.format(m_x, _m_x))
    print('M[Y] = {} : M[Y]* = {}'.format(m_y, _m_y))
    print('D[X] = {} : D[X]* = {}'.format(d_x, _d_x))
    print('D[Y] = {} : D[Y]* = {}'.format(d_y, _d_y))


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


def intervals_estimate_discrete(x_y_matrix, distribution_table, n, q):
    _m_x = me_disc(x_y_matrix, True)
    _m_y = me_disc(x_y_matrix, False)
    _d_x = disp_disc(x_y_matrix, _m_x, True)
    _d_y = disp_disc(x_y_matrix, _m_y, False)
    _s_x = sqrt(_d_x)
    _s_y = sqrt(_d_y)

    k_x = norm.ppf(q) * _s_x / sqrt(n)
    k_y = norm.ppf(q) * _s_y / sqrt(n)

    print('\nConfidence interval for ME (quantile - {}):'.format(q))
    print('X: {} < M < {}'.format(_m_x - k_x, _m_x + k_x))
    print('Y: {} < M < {}'.format(_m_y - k_y, _m_y + k_y))

    print('\nConfidence interval for Dispersion (quantile - {}):'.format(q))
    excess_x, excess_y = excess_disc(distribution_table, x_y_matrix, n)
    k_x = norm.ppf(q) * sqrt((excess_x + 2) / n) * _d_x
    k_y = norm.ppf(q) * sqrt((excess_y + 2) / n) * _d_y

    print('X: {} < D < {}'.format(_d_x - k_x, _d_x + k_x))
    print('Y: {} < D < {}'.format(_d_y - k_y, _d_y + k_y))


def excess_disc(distribution_table, x_y_matrix, n):
    n_size = len(distribution_table)
    m_size = len(distribution_table[0])

    m_x = me_disc(distribution_table, True)
    m_y = me_disc(distribution_table, False)
    d_x = disp_disc(distribution_table, m_x, True)
    d_y = disp_disc(distribution_table, m_y, False)

    x_density = []
    for i in range(m_size):
        density = 0

        for j in range(n_size):
            density += x_y_matrix[j][i]
        x_density.append(density * n)

    y_density = [sum(row) * n for row in x_y_matrix]

    mu_x = sum([(value - m_x) ** 4 * x_density[value] for value in range(m_size)]) / n
    mu_y = sum([(value - m_y) ** 4 * y_density[value] for value in range(n_size)]) / n

    return mu_x / d_x ** 2 - 3, mu_y / d_y ** 2 - 3


def check_correlation_discrete(x_y_matrix):
    _m_x = me_disc(x_y_matrix, True)
    _m_y = me_disc(x_y_matrix, False)
    _d_x = disp_disc(x_y_matrix, _m_x, True)
    _d_y = disp_disc(x_y_matrix, _m_y, False)

    covariance = 0
    for i in range(len(x_y_matrix)):
        for j in range(len(x_y_matrix[0])):
            covariance += (i - _m_x) * (i - _m_y) * x_y_matrix[i][j]

    correlation = covariance / sqrt(_d_x * _d_y)
    print('\nCorrelation: {}'.format(correlation))


def z_test_discrete(distribution_table, x_y_matrix, n):
    _m_x = me_disc(x_y_matrix, True)
    _m_y = me_disc(x_y_matrix, False)
    m_x = me_disc(distribution_table, True)
    m_y = me_disc(distribution_table, False)
    d_x = disp_disc(distribution_table, m_x, True)
    d_y = disp_disc(distribution_table, m_y, False)

    z_x = (_m_x - m_x) / sqrt(d_x * n)
    z_y = (_m_y - m_y) / sqrt(d_y * n)

    print('\nZ-test:')
    print('X: {}'.format(z_x))
    print('Y: {}'.format(z_y))


def f_test_discrete(distribution_table, x_y_matrix):
    _m_x = me_disc(x_y_matrix, True)
    _m_y = me_disc(x_y_matrix, False)
    _d_x = disp_disc(x_y_matrix, _m_x, True)
    _d_y = disp_disc(x_y_matrix, _m_y, False)
    m_x = me_disc(distribution_table, True)
    m_y = me_disc(distribution_table, False)
    d_x = disp_disc(distribution_table, m_x, True)
    d_y = disp_disc(distribution_table, m_y, False)

    if _d_x > d_x:
        f_x = _d_x / d_x
    else:
        f_x = d_x / _d_x

    if _d_y > d_y:
        f_y = _d_y / d_y
    else:
        f_y = d_y / _d_y

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


def discrete_research(x_y_matrix, n):
    distribution_table = probability_matrix()
    point_estimate_discrete(distribution_table, x_y_matrix)
    intervals_estimate_discrete(distribution_table, x_y_matrix, n, 0.95)
    check_correlation_discrete(x_y_matrix)

    z_test_discrete(distribution_table, x_y_matrix, n)
    f_test_discrete(distribution_table, x_y_matrix)
