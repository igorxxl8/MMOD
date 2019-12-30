from scipy.stats import beta
from lab1 import rand_basic_value_gen


def get_next(a, b):
    return beta.ppf(rand_basic_value_gen.next_value(), a, b)
