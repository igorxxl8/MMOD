from lab2 import (
    continuous_system_rand_value_gen,
    discrete_system_rand_value_gen
)
from lab2 import system_stat_research

N = 100000

if __name__ == '__main__':
    print("\nContinuous system random value generation")
    random_values_system = [continuous_system_rand_value_gen.next_value_x_y() for _ in range(N)]
    system_stat_research.continuous_research(random_values_system)

    print("\nDiscrete system random value generation")
    random_values_system = discrete_system_rand_value_gen.get_empirical_matrix(N)
    system_stat_research.discrete_research(random_values_system, N)
