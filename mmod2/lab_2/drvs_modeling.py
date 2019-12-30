from lab_1 import rbv_generator
from lab_2.drvs_distribution import get_distribution_table


def get_empirical_matrix(n):
    distribution_table = get_distribution_table()
    n_size = len(distribution_table)
    m_size = len(distribution_table[0])

    vector_q = [sum(row) for row in distribution_table]
    vector_l = get_column_intervals(vector_q, n_size)
    vector_r = get_row_intervals(distribution_table, n_size, m_size)
    empirical_matrix = [[0 for _ in range(m_size)] for _ in range(n_size)]

    for i in range(n):
        random_value = rbv_generator.get_next()

        for k in range(1, len(vector_l)):
            if vector_l[k - 1] < random_value <= vector_l[k]:
                is_found_second_component = False

                while not is_found_second_component:
                    random_value = rbv_generator.get_next()

                    for s in range(1, len(vector_r[k - 1])):
                        if vector_r[k - 1][s - 1] < random_value <= vector_r[k - 1][s]:
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
