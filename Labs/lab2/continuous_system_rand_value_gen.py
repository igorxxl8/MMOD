from lab1 import rand_basic_value_gen
from lab2.continuous_system_func import inv_func_x, inv_func_y


def next_value_x_y():
    val1 = rand_basic_value_gen.next_value()
    val2 = rand_basic_value_gen.next_value()
    X = inv_func_x(val1)
    return X, inv_func_y(val2, X)
