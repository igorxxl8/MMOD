import numpy as np
from math import log, factorial
from statistics import mean

number_of_channels = 0
queue_size = 0
request_intensity = 0
service_intensity = 0
probability_to_serve_the_request = 0

service_channels = []
queue = []

done_requests = []
done_requests_ids = []
rejected_requests = []
cancelled_requests = []

current_time = 0
request_number = 0

time_interval = 0.01
time_for_new_request = 0

channels_and_queue_state = []
channels_state = []
queue_state = []


def start(n, m, lambda_, mu, p, run_life_time):
    global current_time, time_for_new_request

    init(n, m, lambda_, mu, p)
    create_new_request()
    time_for_new_request = request_time_generator() + current_time

    while current_time <= run_life_time:
        current_time += time_interval
        process_queue()
        process_channels()

        get_statistic()

        if time_for_new_request <= current_time:
            create_new_request()
            time_for_new_request = request_time_generator() + current_time

    show_statistic()


def create_new_request():
    global request_number
    request_number += 1

    request_id = request_number
    if len(queue) < queue_size:
        queue_request(request_id)
    else:
        reject_request(request_id)


def queue_request(request_id):
    queue.append((request_id, current_time))


def process_queue():
    free_channel_index = find_free_channel()

    if len(queue) == 0:
        return

    if free_channel_index == -1:
        return

    request_id, start_time = queue.pop(0)
    time_in_queue = current_time - start_time
    serve_request(request_id, start_time, time_in_queue, free_channel_index)


def process_channels():
    for channel in service_channels:
        if channel is None:
            continue

        if channel[5] <= current_time:
            accept_request(channel)


def serve_request(request_id, start_time, time_in_queue, free_channel_index):
    serve_time = serve_time_generator()
    end_time = serve_time + current_time
    service_channels[free_channel_index] = \
        (request_id, start_time, time_in_queue, free_channel_index, serve_time, end_time)


def find_free_channel():
    if service_channels.__contains__(None):
        return service_channels.index(None)

    return -1


def accept_request(request):
    free_channel(request[3])

    is_accept = np.random.choice([True, False],
                                 p=[probability_to_serve_the_request, 1 - probability_to_serve_the_request])

    if is_accept:
        done_request(request)
    else:
        cancel_request(request[0])
        try_to_process_request(request[0])


def done_request(request):
    time_in_queuing_system = current_time - request[1]
    done_requests.append((request[0], request[2], request[4], time_in_queuing_system))


def reject_request(request_id):
    rejected_requests.append(request_id)


def cancel_request(request_id):
    cancelled_requests.append(request_id)


def free_channel(index):
    service_channels[index] = None


def try_to_process_request(request_id):
    if len(queue) < queue_size:
        queue_request(request_id)
    else:
        reject_request(request_id)


def init(n, m, lambda_, mu, p):
    global number_of_channels, queue_size, request_intensity, \
        service_intensity, service_channels, probability_to_serve_the_request

    number_of_channels = n
    queue_size = m
    request_intensity = lambda_
    service_intensity = mu
    service_channels = [None for _ in range(n)]
    probability_to_serve_the_request = p


def request_time_generator():
    return exponential_value_generator(request_intensity)


def serve_time_generator():
    return exponential_value_generator(service_intensity)


def exponential_value_generator(intensity):
    return - 1 / intensity * log(1 - np.random.uniform())


def get_final_probabilities():
    final_probabilities = []
    n = number_of_channels
    m = queue_size
    lambda_ = request_intensity
    mu = service_intensity
    p = probability_to_serve_the_request

    p0 = (sum([lambda_ ** i / (factorial(i) * (mu * p) ** i) for i in range(n + 1)]) +
          sum([lambda_ ** i / (factorial(n) * n ** (i - n) * (mu * p) ** i) for i in range(n + 1, n + m)]) +
          lambda_ ** (n + m) / (factorial(n) * n ** m * mu ** (n + m) * p ** (n + m - 1))) ** -1

    final_probabilities.append(p0)

    for i in range(1, n + 1):
        p_ = lambda_ ** i / (factorial(i) * (mu * p) ** i) * p0
        final_probabilities.append(p_)

    for i in range(n + 1, n + m):
        p_ = lambda_ ** i / (factorial(n) * n ** (i - n) * (mu * p) ** i) * p0
        final_probabilities.append(p_)

    p_ = lambda_ ** (n + m) / (factorial(n) * n ** m * mu ** (n + m) * p ** (n + m - 1)) * p0
    final_probabilities.append(p_)

    return final_probabilities


def get_average_number_of_request_in_queue(final_probabilities):
    n = number_of_channels
    m = queue_size

    return sum([i * final_probabilities[n + i] for i in range(1, m + 1)])


def get_statistic():
    n = number_of_channels
    channels_and_queue_state.append(n - service_channels.count(None) + len(queue))
    channels_state.append(n - service_channels.count(None))
    queue_state.append(len(queue))


def get_empirical_final_probabilities():
    empirical_final_probabilities = []

    n = number_of_channels
    m = queue_size

    n_ = len(channels_and_queue_state)
    for i in range(n + m + 1):
        p = channels_and_queue_state.count(i) / n_
        empirical_final_probabilities.append(p)

    return empirical_final_probabilities


def get_empirical_average_service_request_time():
    average_service_request_time = []

    for i in range(len(done_requests)):
        average_service_request_time.append(done_requests[i][2])

    return mean(average_service_request_time)


def get_empirical_average_queue_time():
    average_queue_time = []

    for i in range(len(done_requests)):
        average_queue_time.append(done_requests[i][1])

    return mean(average_queue_time)


def get_empirical_average_request_time_in_system():
    average_request_time_in_system = []

    for i in range(len(done_requests)):
        average_request_time_in_system.append(done_requests[i][3])

    return mean(average_request_time_in_system)


def get_relative_bandwidth():
    return len(done_requests) / (len(done_requests) + len(rejected_requests))


def get_average_channel_usage(final_probabilities):
    n = number_of_channels
    m = queue_size

    return (sum([i * final_probabilities[i] for i in range(n + 1)]) +
            sum([n * final_probabilities[n + i] for i in range(1, m + 1)]))


def show_statistic():
    final_probabilities = get_final_probabilities()
    probability_of_idle_channels = final_probabilities[0]
    denial_of_service_probability = final_probabilities[-1]
    relative_bandwidth = 1 - denial_of_service_probability
    absolute_bandwidth = request_intensity * relative_bandwidth
    average_channel_usage = get_average_channel_usage(final_probabilities)
    average_number_of_request_in_queue = get_average_number_of_request_in_queue(final_probabilities)
    average_number_of_request_in_system = average_channel_usage + average_number_of_request_in_queue
    average_service_request_time = average_channel_usage / request_intensity
    average_queue_time = average_number_of_request_in_queue / request_intensity
    average_request_time_in_system = average_service_request_time + average_queue_time

    final_probabilities_empirical = get_empirical_final_probabilities()
    probability_of_idle_channels_empirical = final_probabilities_empirical[0]
    denial_of_service_probability_empirical = final_probabilities_empirical[-1]
    relative_bandwidth_empirical = 1 - denial_of_service_probability_empirical
    absolute_bandwidth_empirical = request_intensity * relative_bandwidth_empirical
    average_channel_usage_empirical = mean(channels_state)
    average_number_of_request_in_queue_empirical = mean(queue_state)
    average_number_of_request_in_system_empirical = mean(channels_and_queue_state)
    average_service_request_time_empirical = get_empirical_average_service_request_time()
    average_queue_time_empirical = get_empirical_average_queue_time()
    average_request_time_in_system_empirical = get_empirical_average_request_time_in_system()

    print('Теоритические и эмпирические характеристики:')
    print(f'Финальные вероятности: {final_probabilities} -\n\t\t\t\t\t\t{final_probabilities_empirical}')
    print(f'Вероятность простоя каналов: {probability_of_idle_channels} - {probability_of_idle_channels_empirical}')
    print(f'Вероятность отказа обсуживания: {denial_of_service_probability} - '
          f'{denial_of_service_probability_empirical}')
    print(f'Относительная пропускная способность: {relative_bandwidth} - {relative_bandwidth_empirical}')
    print(f'Абсолютная пропускная способность: {absolute_bandwidth} - {absolute_bandwidth_empirical}')
    print(f'Среднее число занятых каналов: {average_channel_usage} - {average_channel_usage_empirical}')
    print(f'Среднее число заявок в очереди: {average_number_of_request_in_queue} - '
          f'{average_number_of_request_in_queue_empirical}')
    print(f'Среднее число заявок в системе: {average_number_of_request_in_system} - '
          f'{average_number_of_request_in_system_empirical}')
    print(f'Среднее время заявки под обслуживанием: {average_service_request_time} - '
          f'{average_service_request_time_empirical}')
    print(f'Среднее время заявки в очереди: {average_queue_time} - {average_queue_time_empirical}')
    print(f'Среднее время заявки в системе: {average_request_time_in_system} - '
          f'{average_request_time_in_system_empirical}')
