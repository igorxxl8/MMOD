from lab_1 import rbv_generator
from lab_1.disrtibution import *


def get_next(distribution):
    return get_discrete_next(distribution)


def get_discrete_next(distribution):
    basic_random_value = rbv_generator.get_next()

    for interval_and_value in distribution.additional_vector:
        if interval_and_value[0] <= basic_random_value < interval_and_value[1]:
            return interval_and_value[2]


class DiscreteDistribution:
    def __init__(self, distribution, n=None, k=None, p=None, l=None):
        self.name = distribution

        if n is not None:
            self.n = n

        if k is not None:
            self.k = k

        if p is not None:
            self.p = p
            self.q = 1 - p

        if l is not None:
            self.l = l

        if distribution == GEOMETRIC:
            self.discrete_values = [
                probability_function_of_the_geometric_distribution(i, self.p) for i in range(self.n)
            ]
            self.m = self.q / self.p
            self.d = self.q / (self.p ** 2)
            self.probability_function_of_the_distribution = \
                lambda x: probability_function_of_the_geometric_distribution(x, self.p)
            self.function_of_the_distribution = \
                lambda x: function_of_the_geometric_distribution(x, self.p)

        if distribution == POISSON:
            self.discrete_values = [
                probability_function_of_the_poisson_distribution(i, self.l) for i in range(self.n)
            ]
            self.m = self.l
            self.d = l
            self.probability_function_of_the_distribution = \
                lambda x: probability_function_of_the_poisson_distribution(x, self.l)
            self.function_of_the_distribution = \
                lambda x: function_of_the_poisson_distribution(x, self.l)

        self.additional_vector = self.get_additional_vector()

    def get_additional_vector(self):
        probability = [p[1] for p in self.discrete_values]

        additional_vector = [
            (sum(probability[:i - 1]), sum(probability[:i]), i - 1)
            for i in range(1, len(probability) + 1)
        ]

        last_value = additional_vector.pop()
        additional_vector.append((last_value[0], 1, last_value[2]))

        return additional_vector
