M_1 = 2 ** 31 - 1
M_2 = 2 ** 31
SEED_1 = 65539
SEED_2 = 65539
MULTIPLIER_1 = 16807
MULTIPLIER_2 = 65539


def get_next():
    return next(random_value)


def multiplicative_congruential_method(seed, k, m):
    while True:
        seed = (k * seed) % m
        yield seed / m


def mclaren_marsaglia_method(k_size_table):
    first_sequence = multiplicative_congruential_method(SEED_1, MULTIPLIER_1, M_1)
    second_sequence = multiplicative_congruential_method(SEED_2, MULTIPLIER_2, M_1)

    table_v = [next(first_sequence) for _ in range(k_size_table)]

    while True:
        index_s = int(next(second_sequence) * k_size_table)
        value = table_v[index_s]
        table_v[index_s] = next(first_sequence)
        yield value


random_value = mclaren_marsaglia_method(2 ** 8)
