from lab1 import (
    stat_research,
    rand_basic_value_gen,
    continuous_rand_value_method,
    discrete_rand_value_method
)
from lab1.values_funcs import *

Q = 0.99
N = 10000

if __name__ == '__main__':
    print("Random basic value generator")
    M = 1 / 2
    D = 1 / 12
    S = 3

    random_values = [rand_basic_value_gen.next_value() for _ in range(N)]
    stat_research.research(random_values, N, M, D, Q, UNIFORM, s=S)

    print("\nContinuous random value generation")
    a = 2
    b = 4
    M = a / (a + b)
    D = (a * b) / ((a + b) ** 2 * (a + b + 1))

    random_values = [continuous_rand_value_method.get_next(a, b) for _ in range(N)]
    stat_research.research(random_values, N, M, D, Q, BETA, None, a, b)

    print("\nDiscrete random value generation")
    P = 0.5
    L = 5
    K = 20

    distribution = discrete_rand_value_method.DiscreteDistribution(GEOMETRIC, n=K, p=P)

    M = distribution.m
    D = distribution.d

    random_values = [discrete_rand_value_method.get_next(distribution) for _ in range(N)]
    stat_research.discrete_research(random_values, N, K, M, D, Q, distribution)
