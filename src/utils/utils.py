import numpy as np

def compute_target_position_trajectory(t, params):
    t_x = params['trajectory']['t_x']
    k_v = params['trajectory']['k_v']
    k_a = params['trajectory']['k_a']
    epsilon = params['trajectory']['epsilon']

    term1 = t_x / 2
    term2_numerator = np.cosh((2 * k_a * t) / k_v - epsilon)
    term2_denominator = np.cosh((2 * k_a * t) / k_v - epsilon - (2 * k_a * t_x) / k_v ** 2)
    term2_log = np.log(term2_numerator / term2_denominator)
    term2 = (k_v ** 2 / (4 * k_a)) * term2_log

    return term1 + term2


def compute_target_velocity_trajectory(t, params):
    t_x = params['trajectory']['t_x']
    k_v = params['trajectory']['k_v']
    k_a = params['trajectory']['k_a']
    epsilon = params['trajectory']['epsilon']

    term1 = np.tanh(epsilon - 2 * k_a * t / k_v)
    term2 = np.tanh(epsilon - 2 * k_a * t / k_v + 2 * k_a * t_x / k_v ** 2)

    return -k_v * (term1 - term2) / 2


def compute_target_acceleration_trajectory(t, params):
    t_x = params['trajectory']['t_x']
    k_v = params['trajectory']['k_v']
    k_a = params['trajectory']['k_a']
    epsilon = params['trajectory']['epsilon']

    term1 = np.tanh(epsilon - 2 * k_a * t / k_v) ** 2
    term2 = np.tanh(epsilon - 2 * k_a * t / k_v + 2 * k_a * t_x / k_v ** 2) ** 2

    return -k_a * (term1 - term2)