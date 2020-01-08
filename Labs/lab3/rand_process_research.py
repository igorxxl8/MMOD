from math import sqrt
from scipy.stats import norm
from statistics import mean, variance
from statsmodels.api import tsa
from matplotlib import pyplot as plt


def points_estimate(random_process, time_line):
    mean_slices = get_mean_slices(random_process, time_line)
    mean_realizations = get_mean_realization(random_process)
    variance_slices = get_variance_slices(random_process, time_line)
    variance_realizations = get_variance_realization(random_process)

    print('Points estimates:')
    print(f'ME by sections: {mean_slices}')
    print(f'ME by realisations: {mean_realizations}')
    print(f'Dispersion by sections: {variance_slices}')
    print(f'Dispersion by realisation: {variance_realizations}')


def get_mean_slices(random_process, time_line):
    mean_slices = []

    for t in range(len(time_line)):
        t_slice = []

        for y_j in random_process:
            t_slice.append((y_j[t]))

        mean_slices.append(mean(t_slice))

    return mean(mean_slices)


def get_mean_realization(random_process):
    return mean(random_process[0])


def get_variance_slices(random_process, time_line):
    variance_slices = []

    for t in range(len(time_line)):
        t_slice = []

        for y_j in random_process:
            t_slice.append((y_j[t]))

        variance_slices.append(variance(t_slice))

    return mean(variance_slices)


def get_variance_realization(random_process):
    return variance(random_process[0])


def intervals_estimate(random_process, time_line, q):
    n = len(time_line)

    mean_slices = get_mean_slices(random_process, time_line)
    variance_slices = get_variance_slices(random_process, time_line)
    s_slices = sqrt(variance_slices)

    print('\nIntervals estimates:')
    k = norm.ppf(q) * s_slices / sqrt(n)

    print(f'Confidence interval for ME:')
    print(f'{mean_slices - k} < M < {mean_slices + k}')

    print(f'Confidence interval for Dispersion:')
    kurtosis = get_excess(random_process, time_line, n)
    k = norm.ppf(q) * sqrt((kurtosis + 2) / n) * variance_slices

    print(f'{variance_slices - k} < D < {variance_slices + k}')


def get_excess(random_process, time_line, n):  # экцесса
    excess_t = []

    for t in range(n):
        t_slice = []

        for y_j in random_process:
            t_slice.append(y_j[t])

        m = mean(t_slice)
        d = variance(t_slice)
        mu = sum([(value - m) ** 4 for value in t_slice])
        excess_t.append(mu / d ** 2 - 3)

    return mean(excess_t)


def dickey_fuller_test(random_process):
    test = tsa.adfuller(random_process[0])

    print('\nDiki-Fuller\'s Test:')
    print(f'Value: {test[0]}')
    print(f'Critical value: {test[4]["5%"]}')

    if test[0] > test[4]['5%']:
        print('The process isn\'t stationary')
    else:
        print('The process is stationary')


def build_spectrum_with_means(random_process):
    fig, ax = plt.subplots(nrows=5, ncols=2)
    t = 0

    for row in ax:
        for col in row:
            col.plot(random_process[t])
            col.set_title(f'Realisation #{t}')
            t += 1

    plt.show()


def do_research(random_process, time_line):
    q = 0.95
    build_histogram(random_process[0])
    points_estimate(random_process, time_line)
    build_spectrum_with_means(random_process)
    intervals_estimate(random_process, time_line, q)
    dickey_fuller_test(random_process)
    slutsky_test(random_process)


def build_histogram(random_process_real):
    plt.hist(random_process_real, density=True)
    plt.show()


def slutsky_test(random_process):
    print('\nSlutskii condition:')
    T = len(random_process)
    integral = 0

    for t in range(T):
        integral += correlation(random_process, t)

    print(f'Value: {integral / T} -> 0')


def correlation(random_process, tau):
    arr = []

    for t in range(len(random_process) - tau):
        y, y_ = random_process[t], random_process[t + tau]
        temp, T = 0, len(y)

        for j in range(T):
            temp += (y[j] * y_[j])

        temp /= T
        arr.append(temp)

    return mean(arr)
