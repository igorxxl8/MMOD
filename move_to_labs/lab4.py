from lab4 import queuing_system_modeling

if __name__ == '__main__':
    n = 2
    m = 2
    request_intensity = 1.5  # lambda
    service_intensity = 1  # mu
    time_interval_for_info = 2
    p = 0.5

    run_life_time = 1000

    queuing_system_modeling.start(
        n,
        m,
        request_intensity,
        service_intensity,
        p,
        run_life_time
    )
