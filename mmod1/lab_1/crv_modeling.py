from math import sqrt, log, sin, cos, pi

from lab_1 import rbv_generator
from lab_1.disrtibution import density_of_the_normal_distribution

N = 12


def get_next(m, sigma):
    return first_algorithm(m, sigma)


def get_next_s(m, sigma):
    return second_algorithm(m, sigma)


def get_next_n(m, sigma):
    return neumann_method(m, sigma)


def first_algorithm(m, sigma):
    return m + sigma * (sqrt(N / 12) * sum([rbv_generator.get_next() for _ in range(N)]) - N / 2)


def second_algorithm(m, sigma):
    a1 = rbv_generator.get_next()
    a2 = rbv_generator.get_next()
    n1 = m + sigma * sqrt(-2 * log(a1)) * cos(2 * pi * a2)
    n2 = m + sigma * sqrt(-2 * log(a1)) * sin(2 * pi * a2)

    return n1, n2


def neumann_method(m, sigma):
    while True:
        a = -sigma * 3 + m
        b = sigma * 3 + m

        f_max = density_of_the_normal_distribution(m, m, sigma)
        x1 = a + (b - a) * rbv_generator.get_next()
        x2 = rbv_generator.get_next() * f_max

        if x2 <= density_of_the_normal_distribution(x1, m, sigma):
            return x1
        else:
            continue
