import numpy as np
from mushroom.utils.angles import shortest_angular_distance


def angle_to_angle_diff_complete_state(inputs):
    alpha_ref = inputs[0]
    states = inputs[1]
    alpha = states[1]
    alpha_dot = states[2]
    beta_dot = states[3]
    delta_alpha = shortest_angular_distance(alpha_ref[0], alpha)

    return np.array([delta_alpha, alpha_dot, beta_dot])



