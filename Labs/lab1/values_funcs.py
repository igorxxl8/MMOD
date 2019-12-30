from math import sqrt, gamma
from scipy.stats import beta


def uniform_distribution_function(x, a, b):
    return (x - a) / (b - a)


def uniform_distribution_density(a, b):
    return 1 / (b - a)


def beta_distribution_function(x, a, b):
    return beta.cdf(x, a, b)


def _B(a, b):
    return gamma(a) * gamma(b) / gamma(a + b)


def beta_distribution_density(x, a, b):
    return x ** (a - 1) * (1 - x) ** (b - 1) / _B(a, b)


def geometric_distribution_probability_function(x, p):
    q = 1 - p
    return x, (q ** x) * p


def geometric_distribution_function(x, p):
    q = 1 - p
    return x, 1 - q ** (x + 1)


UNIFORM = 'uniform'
GEOMETRIC = 'geometric'
BETA = 'beta'
