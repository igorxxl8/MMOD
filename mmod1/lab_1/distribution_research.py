from math import log10
from scipy.stats import t, chi2
from matplotlib import pyplot as plt
import numpy as np

from lab_1.disrtibution import *


def build_histogram(data, bins=10):
    plt.hist(data, bins=bins)
    plt.show()


def build_histogram_with_density(data, distribution, m, sigma, bins=10):
    x = sorted(data)
    y = []

    if distribution == UNIFORM:
        y = [density_of_the_uniform_distribution(0, 1) for _ in x]
    elif distribution == NORMAL:
        y = [density_of_the_normal_distribution(value, m, sigma) for value in x]

    plt.hist(data, bins=bins, density=True)
    plt.plot(x, y)
    plt.show()


def build_probability_plots(data, distribution, bins=10):
    data = sorted(data)
    n = len(data)

    x = range(bins)
    y1 = [data.count(i) / n for i in x]
    y2 = []

    if distribution.name == GEOMETRIC:
        y2 = [distribution.probability_function_of_the_distribution(value)[1] for value in x]
    elif distribution.name == POISSON:
        y2 = [distribution.probability_function_of_the_distribution(value)[1] for value in x]

    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.legend(['Эмпирический', 'Теоретический'])
    plt.show()


def get_empirical_frequency_values(data, n, k, is_discrete):
    _data = sorted(data)
    segments = [[] for _ in range(k)]

    a = round(_data[0])
    b = round(_data[n - 1])
    interval = (b - a) / k
    start = a
    end = a + interval

    if is_discrete:
        interval = 1
        end = a + interval

    index = 0

    for value in _data:
        if start <= value < end:
            segments[index].append(value)
        elif value < a or value > b:
            segments[index].append(value)
        else:
            start += interval
            end += interval
            index += 1

    p = [len(segment) / n for segment in segments]

    return p


def get_theoretical_frequency_values(data, k, m, sigma, distribution):
    p = []

    a = min(data)
    b = max(data)
    interval = (b - a) / k

    start = a
    end = a + interval

    if distribution == UNIFORM:
        for i in range(k):
            p.append(function_of_the_uniform_distribution(end, 0, 1) -
                     function_of_the_uniform_distribution(start, 0, 1))
            start = end
            end += interval
    elif distribution == NORMAL:
        for i in range(k):
            p.append(function_of_the_normal_distribution(end, m, sigma) -
                     function_of_the_normal_distribution(start, m, sigma))
            start = end
            end += interval
    elif distribution.name == GEOMETRIC:
        interval = 1
        end = a + interval

        p.append(distribution.function_of_the_distribution(start)[1])

        for i in range(k):
            p.append(distribution.function_of_the_distribution(end)[1] -
                     distribution.function_of_the_distribution(start)[1])
            start = end
            end += interval

        p.pop()

    elif distribution.name == POISSON:
        interval = 1
        end = a + interval

        for i in range(k):
            p.append(distribution.function_of_the_distribution(end)[1] -
                     distribution.function_of_the_distribution(start)[1])
            start = end
            end += interval

    return p


def points_estimate(data, n, m, d):
    _m = sum(data) / n
    _d = sum([(value - _m) ** 2 for value in data]) / (n - 1)

    print('Точечные оценки:')
    print('M = {} : M* = {}'.format(m, _m))
    print('D = {} : D* = {}'.format(d, _d))


def points_estimate_discrete(data, n, k, m, d):
    data = sorted(data)
    values = range(k)

    p_empirical = [data.count(i) / n for i in values]

    _m = sum([i * p_empirical[i] for i in values])
    _d = sum([((i - _m) ** 2) * p_empirical[i] for i in values])

    print('Точечные оценки:')
    print('M = {} : M* = {}'.format(m, _m))
    print('D = {} : D* = {}'.format(d, _d))

    return _m, _d


def intervals_estimate(data, n, m, d, q):
    _m = sum(data) / n
    _d = sum([(value - _m) ** 2 for value in data]) / (n - 1)
    s = sqrt(_d)

    k = s * t.ppf(q, n - 1) / sqrt(n - 1)
    print('\nДоверительный интервал для МО (квантиль - {}):'.format(q))
    print('{} <= {} < {}'.format(_m - k, m, _m + k))

    k1 = n * _d / chi2.isf((1 - q) / 2, n - 1)
    k2 = n * _d / chi2.isf((1 + q) / 2, n - 1)
    print('\nДоверительный интервал для дисперсии (квантиль - {}):'.format(q))
    print('{} <= {} < {}'.format(k1, d, k2))


def intervals_estimate_discrete(n, m, d, _m, _d, q):
    s = sqrt(_d)

    k = s * t.ppf(q, n - 1) / sqrt(n - 1)
    print('\nДоверительный интервал для МО (квантиль - {}):'.format(q))
    print('{} <= {} < {}'.format(_m - k, m, _m + k))

    k1 = n * _d / chi2.isf((1 - q) / 2, n - 1)
    k2 = n * _d / chi2.isf((1 + q) / 2, n - 1)
    print('\nДоверительный интервал для дисперсии (квантиль - {}):'.format(q))
    print('{} <= {} < {}'.format(k1, d, k2))


def check_correlation(data, m, d, n, s):
    m_x_y = []

    for i in range(n - s):
        m_x_y.append((data[i], data[i + s]))

    r = ((sum([x * y for (x, y) in m_x_y]) / (n - s)) - m * m) / sqrt(d * d)

    print('\nКорреляция:')
    print('R* = {}'.format(r))


def get_empirical_function(data, n):
    values_set = set()

    for value in data:
        values_set.add(value)

    counts = [data.count(i) for i in values_set]
    empirical_function = [([sum(counts[:i + 1]) / n] * counts[i]) for i in range(len(counts))]
    empirical_function_data = []

    for ef in empirical_function:
        for value in ef:
            empirical_function_data.append(value)

    return empirical_function_data


def pearson_criterion(data, n, k, q, distribution, m=None, sigma=None, is_discrete=False):
    p = get_theoretical_frequency_values(data, k, m, sigma, distribution)
    p_star = get_empirical_frequency_values(data, n, k, is_discrete)

    print('\nКритерий хи-квадрат Пирсона:')

    chi_2 = chi2.ppf(q, k - 1)
    chi_2_star = n * sum([((p[i] - p_star[i]) ** 2) / p[i] for i in range(k)])

    if chi_2_star < chi_2:
        print('{} < {}'.format(chi_2_star, chi_2))
        print('Нет оснований отклонять выдвинутую гипотезу')
    else:
        print('{} > {}'.format(chi_2_star, chi_2))
        print('Отклоняем выдвинутую гипотезу')


def kolmogorov_criterion(data, n, q, distribution, m=None, sigma=None):
    max_diff = 0
    f_empirical = get_empirical_function(data, n)
    f_theoretical = None

    if distribution == UNIFORM:
        f_theoretical = [function_of_the_uniform_distribution(value, 0, 1) for value in sorted(data)]
    elif distribution == NORMAL:
        f_theoretical = [function_of_the_normal_distribution(value, m, sigma) for value in sorted(data)]
    elif distribution.name:
        f_theoretical = [distribution.function_of_the_distribution(value)[1] for value in sorted(data)]

    print('\nКритерий Колмлгорова:')

    for i in range(n):
        diff = abs(f_empirical[i] - f_theoretical[i])

        if diff > max_diff:
            max_diff = diff

    kolmogorov = 1.63  # for q = 0.01 (P{sqrt(n) * D > Kt} = 0.01)
    kolmogorov_star = sqrt(n) * max_diff

    if kolmogorov_star < kolmogorov:
        print('{} < {}'.format(kolmogorov_star, kolmogorov))
        print('Нет оснований отклонять выдвинутую гипотезу')
    else:
        print('{} > {}'.format(kolmogorov_star, kolmogorov))
        print('Отклоняем выдвинутую гипотезу')


def do_research_continues(data, n, m, d, q, distribution, s=None):
    k = int(sqrt(n)) if n <= 100 else int(4 * log10(n))
    sigma = sqrt(d)

    print('\nПроверка гипотезы о соответствии закона распределения:')

    build_histogram(data, k)
    build_histogram_with_density(data, distribution, m, sigma, k)
    points_estimate(data, n, m, d)
    intervals_estimate(data, n, m, d, q)

    if distribution == UNIFORM:
        print('\nЧастотный анализ:')
        print(np.array(get_empirical_frequency_values(data, n, k, is_discrete=False)), end=' -> {}'.format(1 / k))
        check_correlation(data, m, d, n, s)

    pearson_criterion(data, n, k, q, distribution, m, sigma)
    kolmogorov_criterion(data, n, q, distribution, m, sigma)


def do_research_discrete(data, n, k, m, d, q, distribution):
    print('\nПроверка гипотезы о соответствии закона распределения:')

    build_probability_plots(data, distribution, k)
    _m, _d = points_estimate_discrete(data, n, k, m, d)
    intervals_estimate_discrete(n, m, d, _m, _d, q)

    pearson_criterion(data, n, k, q, distribution, is_discrete=True)
    kolmogorov_criterion(data, n, q, distribution)
