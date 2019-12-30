from math import log10
from scipy.stats import t, chi2, norm
from matplotlib import pyplot as plt
import numpy as np


from lab1.values_funcs import *


def build_histogram_with_density(data, distribution, m, d, bins=10, extra=None):
    x = sorted(data)
    y = []

    if distribution == UNIFORM:
        y = [uniform_distribution_density(0, 1) for _ in x]
    elif distribution == BETA:
        y = [beta_distribution_density(_, extra["a"], extra["b"]) for _ in x]

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

    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.legend(['Empirical', 'Theoretical'])
    plt.show()


def empirical_frequency_values(data, n, k, is_discrete):
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


def theoretical_frequency_values(data, k, m, sigma, distribution, extra=None):
    p = []

    a = min(data)
    b = max(data)
    interval = (b - a) / k

    start = a
    end = a + interval

    if distribution == UNIFORM:
        for i in range(k):
            p.append(uniform_distribution_function(end, 0, 1) -
                     uniform_distribution_function(start, 0, 1))
            start = end
            end += interval

    elif distribution == BETA:
        for i in range(k):
            p.append(beta_distribution_function(end, extra["a"], extra["b"]) -
                     beta_distribution_function(start, extra["a"], extra["b"]))
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

    return p


def points_estimate(data, n, m, d):
    _m = sum(data) / n
    _d = sum([(value - _m) ** 2 for value in data]) / (n - 1)

    print('Point estimates:')
    print('M = {} : M* = {}'.format(m, _m))
    print('D = {} : D* = {}'.format(d, _d))


def points_estimate_discrete(data, n, k, m, d):
    data = sorted(data)
    values = range(k)

    p_empirical = [data.count(i) / n for i in values]

    _m = sum([i * p_empirical[i] for i in values])
    _d = sum([((i - _m) ** 2) * p_empirical[i] for i in values])

    print('Point estimates:')
    print('M = {} : M* = {}'.format(m, _m))
    print('D = {} : D* = {}'.format(d, _d))

    return _m, _d


def intervals_estimate(data, n, m, d, q):
    _m = sum(data) / n
    _d = sum([(value - _m) ** 2 for value in data]) / (n - 1)
    _m4 = sum([(value - _m) ** 4 for value in data]) / n
    s = sqrt(_d)

    k = s * norm.ppf(q) / sqrt(n)
    print('\nConfidence interval for ME (quantile - {}):'.format(q))
    print('{} <= M < {}'.format(_m - k, _m + k))

    ksi = (_m4 / (_d ** 2)) - 3
    k1 = _d + norm.ppf(q) * sqrt((ksi + 2) / n) * _d
    k2 = _d - norm.ppf(q) * sqrt((ksi + 2) / n) * _d
    print('\nConfidence interval for Dispersion (quantile - {}):'.format(q))
    print('{} <= D < {}'.format(k2, k1))


def intervals_estimate_discrete(n, m, d, _m, _d, q):
    s = sqrt(_d)

    k = s * t.ppf(q, n - 1) / sqrt(n - 1)
    print('\nConfidence interval for ME (quantile - {}):'.format(q))
    print('{} <= {} < {}'.format(_m - k, m, _m + k))

    k1 = n * _d / chi2.isf((1 - q) / 2, n - 1)
    k2 = n * _d / chi2.isf((1 + q) / 2, n - 1)
    print('\nConfidence interval for Dispersion (quantile - {}):'.format(q))
    print('{} <= {} < {}'.format(k1, d, k2))


def check_correlation(data, m, d, n, s):
    m_x_y = []

    for i in range(n - s):
        m_x_y.append((data[i], data[i + s]))

    #r = ((sum([x * y for (x, y) in m_x_y]) / (n - s)) - m * m) / sqrt(d * d)
    r = 12 * (1 / (n - s)) * sum([x * y for (x, y) in m_x_y]) - 3

    print('\nCorrelation:')
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


def pearson(data, n, k, q, distribution, m=None, sigma=None, is_discrete=False, extra=None):
    p = theoretical_frequency_values(data, k, m, sigma, distribution, extra=extra)
    _p = empirical_frequency_values(data, n, k, is_discrete)

    print('\nPearson chi-squared test:')
    square_chi = chi2.ppf(q, k - 1)
    _square_chi = n * sum([((p[i] - _p[i]) ** 2) / p[i] for i in range(k)])

    if _square_chi < square_chi:
        print('{} < {}'.format(_square_chi, square_chi))
        print('There is no reason to reject the hypothesis')
    else:
        print('{} > {}'.format(_square_chi, square_chi))
        print('Reject the hypothesis')


def kolmogorov(data, n, q, distribution, m=None, sigma=None, extra=None):
    max_diff = 0
    f_e = get_empirical_function(data, n)
    f_t = None

    if distribution == UNIFORM:
        f_t = [uniform_distribution_function(value, 0, 1) for value in sorted(data)]
    elif distribution == BETA:
        f_t = beta_distribution_function(sorted(data), extra["a"], extra["b"])
    elif distribution.name:
        f_t = [distribution.function_of_the_distribution(value)[1] for value in sorted(data)]

    print('\nKolmogorov criterion:')
    for i in range(n):
        diff = abs(f_e[i] - f_t[i])

        if diff > max_diff:
            max_diff = diff

    kolmogorov = 1.63
    _kolmogorov = sqrt(n) * max_diff

    if _kolmogorov < kolmogorov:
        print('{} < {}'.format(_kolmogorov, kolmogorov))
        print('There is no reason to reject the hypothesis')
    else:
        print('{} > {}'.format(_kolmogorov, kolmogorov))
        print('Reject the hypothesis')


def research(data, n, m, d, q, distribution, s=None, a=None, b=None):
    k = int(sqrt(n)) if n <= 100 else int(4 * log10(n))
    sigma = sqrt(d)

    print('\nVerification of the hypothesis of compliance with the distribution law:')
    extra = {"a": a, "b": b}
    if distribution == BETA:
        build_histogram_with_density(data, distribution, m, d, k, extra)

    else:
        build_histogram_with_density(data, distribution, m, d, k)

    points_estimate(data, n, m, d)
    intervals_estimate(data, n, m, d, q)

    if distribution == UNIFORM:
        print('\nFrequency analysis:')
        print(np.array(empirical_frequency_values(data, n, k, is_discrete=False)), end=' -> {}'.format(1 / k))
        check_correlation(data, m, d, n, s)

    pearson(data, n, k, q, distribution, m, sigma, extra=extra)
    kolmogorov(data, n, q, distribution, m, sigma, extra)


def discrete_research(data, n, k, m, d, q, distribution):
    print('\nVerification of the hypothesis of compliance with the distribution law:')

    build_probability_plots(data, distribution, k)
    _m, _d = points_estimate_discrete(data, n, k, m, d)
    intervals_estimate_discrete(n, m, d, _m, _d, q)

    pearson(data, n, k, q, distribution, is_discrete=True)
    kolmogorov(data, n, q, distribution)
