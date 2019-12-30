from math import sin, cos, acos, sqrt


def inv_func_x(x):
    return acos(0.5 * (sqrt(-4 * (x ** 2) + 4 * x + 1) - 2 * x + 1))


def inv_func_y(y, x):
    return -x + acos(-y*sin(x) - y * cos(x) + cos(x))


def x_m():
    return 0.78539


def y_m():
    return 0.78539


def x_d():
    return 0.187647


def y_d():
    return 0.187647
