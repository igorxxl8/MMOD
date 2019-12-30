from math import exp, log


def get_density_both(x, y):
    return exp(-x-y)


def get_density_x(x):
    return exp(-x)


def get_density_y(y):
    return exp(-y)


def get_function_both(x, y):
    return (1 - exp(-x)) * (1 - exp(-y))


def get_function_x(x):
    return 1 - exp(-x)


def get_function_y(y):
    return 1 - exp(-y)


def get_inverse_function_x(x):
    return log(1 / (1 - x))


def get_inverse_function_y(y):
    return log(1 / (1 - y))


def get_x_m():  # integrate x * exp(-x)dx, x=0,oo
    return 1


def get_y_m():
    return 1


def get_x_d():  # integrate (x - m)^2 * exp(-x)dx, x=0,oo
    return 1


def get_y_d():
    return 1
