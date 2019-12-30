M = 2 ** 32
X = 1
A = 1664525
C = 1013904223


def next_value():
    return next(gen)


def linear_congruential_method(x, a, m):
    while True:
        x = (a * x) % m
        yield x / m


gen = linear_congruential_method(X, A, M)
