import numpy as np
from math import sqrt, exp, pi

from lab3.white_noise_modeling import generate_white_noise


def rand_process(time_line, delta_tau, n, sigma_x, sigma_y, alpha):
    x = generate_white_noise(len(time_line))
    y = yt(x, delta_tau, n, sigma_x, sigma_y, alpha)

    return y


def yt(x, delta_tau, n, sigma_x, sigma_y, alpha):
    y = []

    for j in range(len(x)):
        yj = 0

        for i in range(n):
            dt = i * delta_tau
            yj += h(dt, sigma_x, sigma_y, alpha) * x[j - i]

        y.append(delta_tau * yj)

    return y


def h(tau, sigma_x, sigma_y, alpha):
    return sigma_y / sigma_x * sqrt((2 * alpha ** 5 / (3 * pi))) * exp(-alpha * tau) * tau ** 2


def get_time_line(time_size, delta_tau):
    return np.arange(1, time_size, delta_tau)
