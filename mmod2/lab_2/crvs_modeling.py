from lab_1 import rbv_generator
from lab_2.crvs_distribution import get_inverse_function_x, get_inverse_function_y


def get_next_x_y():
    random_value_1 = rbv_generator.get_next()
    random_value_2 = rbv_generator.get_next()

    x = get_inverse_function_x(random_value_1)
    y = get_inverse_function_y(random_value_2)

    return x, y
