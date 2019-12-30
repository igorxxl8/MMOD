from lab1 import rand_basic_value_gen
from lab2.discrete_system_random_values_func import probability_matrix


def get_empirical_matrix(n):
    table = probability_matrix()

    n = len(table)
    m = len(table[0])

    q = [sum(row) for row in table]
    l = get_column_intervals(q, n)
    r = get_row_intervals(table, n, m)
    empirical_matrix = [[0 for _ in range(m)] for _ in range(n)]

    for i in range(n):
        random_value = rand_basic_value_gen.next_value()

        for k in range(1, len(l)):
            if l[k - 1] < random_value <= l[k]:
                is_found_second_component = False

                while not is_found_second_component:
                    random_value = rand_basic_value_gen.next_value()

                    for s in range(1, len(r[k - 1])):
                        if r[k - 1][s - 1] < random_value <= r[k - 1][s]:
                            empirical_matrix[k - 1][s - 1] += 1 / n
                            is_found_second_component = True
                            break
                break

    return empirical_matrix


def get_column_intervals(q, n):
    vector_l = [0]

    for i in range(n):
        vector_l.append(vector_l[i] + q[i - 1])

    return vector_l


def get_row_intervals(distribution_table, n, m):
    vector_r = [[0] for _ in range(n)]

    for i in range(n):
        for j in range(m):
            vector_r[i].append(vector_r[i][j] + distribution_table[i][j])

    return vector_r
