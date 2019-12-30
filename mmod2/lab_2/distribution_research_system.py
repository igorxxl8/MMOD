from matplotlib import pyplot as plt
from math import sqrt, log10
from scipy.stats import norm

from lab_2 import crvs_distribution
from lab_2.drvs_distribution import get_distribution_table


def build_histogram(values, title, bins=10):
    plt.hist(values, density=True, bins=bins)
    plt.title(title)
    plt.show()


def split_system_to_values(system):
    x_values = []
    y_values = []

    for i in range(len(system)):
        x_values.append(system[i][0])
        y_values.append(system[i][1])

    return x_values, y_values


def points_estimate_continues(x_values, y_values):
    n = len(x_values)

    m_x = crvs_distribution.get_x_m()
    m_y = crvs_distribution.get_y_m()
    d_x = crvs_distribution.get_x_d()
    d_y = crvs_distribution.get_y_d()

    m_x_star = get_mean(x_values, n)
    m_y_star = get_mean(y_values, n)
    d_x_star = get_variance(x_values, m_x_star, n)
    d_y_star = get_variance(y_values, m_y_star, n)

    print('Точечные оценки:')
    print(f'M[X] = {m_x} : M[X]* = {m_x_star}')
    print(f'M[Y] = {m_y} : M[Y]* = {m_y_star}')
    print(f'D[X] = {d_x} : D[X]* = {d_x_star}')
    print(f'D[Y] = {d_y} : D[Y]* = {d_y_star}')


def get_mean(values, n):
    return sum(values) / n


def get_variance(values, m, n):
    return sum([(value - m) ** 2 for value in values]) / (n - 1)


def intervals_estimate_continues(x_values, y_values, q):
    n = len(x_values)

    m_x_star = get_mean(x_values, n)
    m_y_star = get_mean(y_values, n)
    d_x_star = get_variance(x_values, m_x_star, n)
    d_y_star = get_variance(y_values, m_y_star, n)
    s_x_star = sqrt(d_x_star)
    s_y_star = sqrt(d_y_star)

    print('\nИнтервальные оценки:')
    k_x = norm.ppf(q) * s_x_star / sqrt(n)
    k_y = norm.ppf(q) * s_y_star / sqrt(n)

    print(f'Доверительный интервал для МО (квантиль - {q}):')
    print(f'X: {m_x_star - k_x} < M < {m_x_star + k_x}')
    print(f'Y: {m_y_star - k_y} < M < {m_y_star + k_y}')

    print(f'Доверительный интервал для дисперсии (квантиль - {q}):')
    kurtosis_x, kurtosis_y = get_kurtosis(x_values, y_values, n)
    k_x = norm.ppf(q) * sqrt((kurtosis_x + 2) / n) * d_x_star
    k_y = norm.ppf(q) * sqrt((kurtosis_y + 2) / n) * d_y_star

    print(f'X: {d_x_star - k_x} < D < {d_x_star + k_x}')
    print(f'Y: {d_y_star - k_y} < M < {d_y_star + k_y}')


def get_kurtosis(x_values, y_values, n):  # экцесса
    m_x = crvs_distribution.get_x_m()
    m_y = crvs_distribution.get_y_m()
    d_x = crvs_distribution.get_x_d()
    d_y = crvs_distribution.get_y_d()

    mu_x = sum([(value - m_x) ** 4 for value in x_values]) / n  # 4-й центральный момент
    mu_y = sum([(value - m_y) ** 4 for value in y_values]) / n

    return mu_x / d_x ** 2 - 3, mu_y / d_y ** 2 - 3


def check_correlation_continues(x_values, y_values):
    n = len(x_values)

    m_x_star = get_mean(x_values, n)
    m_y_star = get_mean(y_values, n)
    d_x_star = get_variance(x_values, m_x_star, n)
    d_y_star = get_variance(y_values, m_y_star, n)

    covariance = sum([(x_values[i] - m_x_star) * (y_values[i] - m_y_star) / n ** 2 for i in range(n)])

    correlation = covariance / sqrt(d_x_star * d_y_star)
    print(f'\nКоэффициент корреляции: {correlation}')


def z_test_continues(x_values, y_values):
    n = len(x_values)

    m_x_star = get_mean(x_values, n)
    m_y_star = get_mean(y_values, n)
    m_x = crvs_distribution.get_x_m()
    m_y = crvs_distribution.get_y_m()
    d_x = crvs_distribution.get_x_d()
    d_y = crvs_distribution.get_y_d()

    z_x = (m_x_star - m_x) / sqrt(d_x * n)
    z_y = (m_y_star - m_y) / sqrt(d_y * n)

    print('\nZ-тест:')
    print(f'X: {z_x}')
    print(f'Y: {z_y}')
    # print('Z:', norm.ppf(0.95) / sqrt(n))


def f_test_continues(x_values, y_values):
    n = len(x_values)

    m_x_star = get_mean(x_values, n)
    m_y_star = get_mean(y_values, n)
    d_x_star = get_variance(x_values, m_x_star, n)
    d_y_star = get_variance(y_values, m_y_star, n)
    d_x = crvs_distribution.get_x_d()
    d_y = crvs_distribution.get_y_d()

    if d_x_star > d_x:
        f_x = d_x_star / d_x
    else:
        f_x = d_x / d_x_star

    if d_y_star > d_y:
        f_y = d_y_star / d_y
    else:
        f_y = d_y / d_y_star

    print('\nF-тест:')
    print(f'X: {f_x}')
    print(f'Y: {f_y}')


def point_estimate_discrete(distribution_table, x_y_matrix):
    m_x = get_mean_discrete(distribution_table, True)
    m_y = get_mean_discrete(distribution_table, False)
    d_x = get_variance_discrete(distribution_table, m_x, True)
    d_y = get_variance_discrete(distribution_table, m_y, False)

    m_x_star = get_mean_discrete(x_y_matrix, True)
    m_y_star = get_mean_discrete(x_y_matrix, False)
    d_x_star = get_variance_discrete(x_y_matrix, m_x_star, True)
    d_y_star = get_variance_discrete(x_y_matrix, m_y_star, False)

    print('Точечные оценки:')
    print(f'M[X] = {m_x} : M[X]* = {m_x_star}')
    print(f'M[Y] = {m_y} : M[Y]* = {m_y_star}')
    print(f'D[X] = {d_x} : D[X]* = {d_x_star}')
    print(f'D[Y] = {d_y} : D[Y]* = {d_y_star}')


def get_mean_discrete(matrix, is_for_x):
    m = 0

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if is_for_x:
                m += matrix[i][j] * i
            else:
                m += matrix[i][j] * j

    return m


def get_variance_discrete(matrix, m, is_for_x):
    d = 0

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if is_for_x:
                d += matrix[i][j] * (i - m) ** 2
            else:
                d += matrix[i][j] * (j - m) ** 2

    return d


def intervals_estimate_discrete(x_y_matrix, distribution_table, n, q):
    m_x_star = get_mean_discrete(x_y_matrix, True)
    m_y_star = get_mean_discrete(x_y_matrix, False)
    d_x_star = get_variance_discrete(x_y_matrix, m_x_star, True)
    d_y_star = get_variance_discrete(x_y_matrix, m_y_star, False)
    s_x_star = sqrt(d_x_star)
    s_y_star = sqrt(d_y_star)

    print('\nИнтервальные оценки:')
    k_x = norm.ppf(q) * s_x_star / sqrt(n)
    k_y = norm.ppf(q) * s_y_star / sqrt(n)

    print(f'Доверительный интервал для МО (квантиль - {q}):')
    print(f'X: {m_x_star - k_x} < M < {m_x_star + k_x}')
    print(f'Y: {m_y_star - k_y} < M < {m_y_star + k_y}')

    print(f'Доверительный интервал для дисперсии (квантиль - {q}):')
    kurtosis_x, kurtosis_y = get_kurtosis_discrete(distribution_table, x_y_matrix, n)
    k_x = norm.ppf(q) * sqrt((kurtosis_x + 2) / n) * d_x_star
    k_y = norm.ppf(q) * sqrt((kurtosis_y + 2) / n) * d_y_star

    print(f'X: {d_x_star - k_x} < D < {d_x_star + k_x}')
    print(f'Y: {d_y_star - k_y} < M < {d_y_star + k_y}')


def get_kurtosis_discrete(distribution_table, x_y_matrix, n):  # экцесса
    n_size = len(distribution_table)
    m_size = len(distribution_table[0])

    m_x = get_mean_discrete(distribution_table, True)
    m_y = get_mean_discrete(distribution_table, False)
    d_x = get_variance_discrete(distribution_table, m_x, True)
    d_y = get_variance_discrete(distribution_table, m_y, False)

    x_density = []
    for i in range(m_size):
        density = 0

        for j in range(n_size):
            density += x_y_matrix[j][i]
        x_density.append(density * n)

    y_density = [sum(row) * n for row in x_y_matrix]

    mu_x = sum([(value - m_x) ** 4 * x_density[value] for value in range(m_size)]) / n  # 4-й центральный момент
    mu_y = sum([(value - m_y) ** 4 * y_density[value] for value in range(n_size)]) / n

    return mu_x / d_x ** 2 - 3, mu_y / d_y ** 2 - 3


def check_correlation_discrete(x_y_matrix):
    m_x_star = get_mean_discrete(x_y_matrix, True)
    m_y_star = get_mean_discrete(x_y_matrix, False)
    d_x_star = get_variance_discrete(x_y_matrix, m_x_star, True)
    d_y_star = get_variance_discrete(x_y_matrix, m_y_star, False)

    covariance = 0
    for i in range(len(x_y_matrix)):
        for j in range(len(x_y_matrix[0])):
            covariance += (i - m_x_star) * (i - m_y_star) * x_y_matrix[i][j]

    correlation = covariance / sqrt(d_x_star * d_y_star)
    print(f'\nКоэффициент корреляции: {correlation}')


def z_test_discrete(distribution_table, x_y_matrix, n):
    m_x_star = get_mean_discrete(x_y_matrix, True)
    m_y_star = get_mean_discrete(x_y_matrix, False)
    m_x = get_mean_discrete(distribution_table, True)
    m_y = get_mean_discrete(distribution_table, False)
    d_x = get_variance_discrete(distribution_table, m_x, True)
    d_y = get_variance_discrete(distribution_table, m_y, False)

    z_x = (m_x_star - m_x) / sqrt(d_x * n)
    z_y = (m_y_star - m_y) / sqrt(d_y * n)

    print('\nZ-тест:')
    print(f'X: {z_x}')
    print(f'Y: {z_y}')
    # print('Z:', norm.ppf(0.95) / sqrt(n))


def f_test_discrete(distribution_table, x_y_matrix):
    m_x_star = get_mean_discrete(x_y_matrix, True)
    m_y_star = get_mean_discrete(x_y_matrix, False)
    d_x_star = get_variance_discrete(x_y_matrix, m_x_star, True)
    d_y_star = get_variance_discrete(x_y_matrix, m_y_star, False)
    m_x = get_mean_discrete(distribution_table, True)
    m_y = get_mean_discrete(distribution_table, False)
    d_x = get_variance_discrete(distribution_table, m_x, True)
    d_y = get_variance_discrete(distribution_table, m_y, False)

    if d_x_star > d_x:
        f_x = d_x_star / d_x
    else:
        f_x = d_x / d_x_star

    if d_y_star > d_y:
        f_y = d_y_star / d_y
    else:
        f_y = d_y / d_y_star

    print('\nF-тест:')
    print(f'X: {f_x}')
    print(f'Y: {f_y}')


def do_research_continues(x_y_system):
    x_values, y_values = split_system_to_values(x_y_system)
    n = len(x_values)
    k = int(sqrt(n)) if n <= 100 else int(4 * log10(n))

    build_histogram(x_values, 'x vector', k)
    build_histogram(y_values, 'y vector', k)

    points_estimate_continues(x_values, y_values)
    intervals_estimate_continues(x_values, y_values, 0.95)
    check_correlation_continues(x_values, y_values)

    z_test_continues(x_values, y_values)
    f_test_continues(x_values, y_values)


def do_research_discrete(x_y_matrix, n):
    distribution_table = get_distribution_table()
    point_estimate_discrete(distribution_table, x_y_matrix)
    intervals_estimate_discrete(distribution_table, x_y_matrix, n, 0.95)
    check_correlation_discrete(x_y_matrix)

    z_test_discrete(distribution_table, x_y_matrix, n)
    f_test_discrete(distribution_table, x_y_matrix)
