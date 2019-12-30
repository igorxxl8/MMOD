from lab1 import rand_basic_value_gen
from lab1.values_funcs import *


def get_next(distribution):
    return get_discrete_next(distribution)


def get_discrete_next(distribution):
    basic_random_value = rand_basic_value_gen.next_value()

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
                geometric_distribution_probability_function(i, self.p) for i in range(self.n)
                if geometric_distribution_probability_function(i, self.p)[1] > 1e-6
            ]

            self.m = self.q / self.p
            self.d = self.q / (self.p ** 2)
            self.probability_function_of_the_distribution = lambda x: geometric_distribution_probability_function(x,
                                                                                                                  self.p)
            self.function_of_the_distribution = lambda x: geometric_distribution_function(x, self.p)

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
