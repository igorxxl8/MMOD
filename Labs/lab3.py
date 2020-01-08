from lab3 import rand_process_modeling
from lab3 import rand_process_research


if __name__ == '__main__':
    sigma_x = 1
    sigma_y = 2
    alpha = 0.5
    delta_tau_max = 8.03 / alpha
    n = 10
    time_size = 10000
    delta_tau = delta_tau_max / n
    k = 10

    random_process = []
    for _ in range(k):
        y = rand_process_modeling.generate_random_process(time_size, delta_tau, n, sigma_x, sigma_y, alpha)
        random_process.append(y)

    time_line = rand_process_modeling.get_time_line(time_size, delta_tau)
    rand_process_research.do_research(random_process, time_line)
