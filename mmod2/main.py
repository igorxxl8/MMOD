from lab_1 import distribution_research, rbv_generator, crv_modeling, drv_modeling
from lab_1.disrtibution import *
from lab_2 import crvs_modeling, drvs_modeling
from lab_2 import distribution_research_system


def lab_1():
    Q = 0.99
    N = 10000

    def a():
        M = 1 / 2
        D = 1 / 12
        S = 3

        random_values = [rbv_generator.get_next() for _ in range(N)]
        distribution_research.do_research_continues(random_values, N, M, D, Q, UNIFORM, s=S)

    def b_first_method():
        M = 0
        D = 1
        SIGMA = sqrt(D)

        random_values = [crv_modeling.get_next(M, SIGMA) for _ in range(N)]
        distribution_research.do_research_continues(random_values, N, M, D, Q, NORMAL)

    def b_second_method():
        M = 0
        D = 1
        SIGMA = sqrt(D)

        normal_random_values = []

        for _ in range(N // 2):
            value_1, value_2 = crv_modeling.get_next_s(M, SIGMA)
            normal_random_values.append(value_1)
            normal_random_values.append(value_2)

        distribution_research.do_research_continues(normal_random_values, N, M, D, Q, NORMAL)

    def b_third_algorithm():
        M = 0
        D = 1
        SIGMA = sqrt(D)

        random_values = [crv_modeling.get_next_n(M, SIGMA) for _ in range(N)]
        distribution_research.do_research_continues(random_values, N, M, D, Q, NORMAL)

    def c():
        P = 0.5
        L = 5
        K = 20

        distribution = drv_modeling.DiscreteDistribution(GEOMETRIC, n=K, p=P)
        # distribution = drv_modeling.DiscreteDistribution(POISSON, n=K, l=L)

        M = distribution.m
        D = distribution.d

        random_values = [drv_modeling.get_next(distribution) for _ in range(N)]
        distribution_research.do_research_discrete(random_values, N, K, M, D, Q, distribution)

    a()
    # b_first_method()
    # b_second_method()
    b_third_algorithm()
    c()


def lab_2():
    N = 10000

    def a():
        random_values_system = [crvs_modeling.get_next_x_y() for _ in range(N)]
        distribution_research_system.do_research_continues(random_values_system)

    def b():
        random_values_system = drvs_modeling.get_empirical_matrix(N)
        distribution_research_system.do_research_discrete(random_values_system, N)
    a()
    b()


if __name__ == '__main__':
    # lab_1()
    lab_2()
