from math import erf, sqrt, pi, exp, factorial


UNIFORM = 'uniform'
NORMAL = 'normal'
GEOMETRIC = 'geometric'
POISSON = 'poisson'


def function_of_the_uniform_distribution(x, a, b):
    return (x - a) / (b - a)


def density_of_the_uniform_distribution(a, b):
    return 1 / (b - a)


def function_of_the_normal_distribution(x, m, sigma):
    return (1 / 2) * (1 + erf((x - m) / sqrt(2 * (sigma ** 2))))


def density_of_the_normal_distribution(x, m, sigma):
    return (1 / (sigma * sqrt(2 * pi))) * exp((-(x - m) ** 2) / (2 * sigma ** 2))


def probability_function_of_the_geometric_distribution(x, p):
    q = 1 - p
    return x, (q ** x) * p


def function_of_the_geometric_distribution(x, p):
    q = 1 - p
    return x, 1 - q ** (x + 1)


def probability_function_of_the_poisson_distribution(x, l):
    return x, exp(-l) * (l ** x) / factorial(x)


def function_of_the_poisson_distribution(x, l):
    return x, exp(-l) * sum([(l ** i) / factorial(i) for i in range(x)])

