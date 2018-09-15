import numpy as np
from mushroom.utils.angles import *


def G_high(inputs):
    reward = inputs[0]
    if reward is not None and reward[0] == 0:
        res = 100.0
    else:
        res = 0.0
    return np.array([res])


def G_low(inputs):
    active_direction = inputs[0]
    if active_direction < 4:
        goal_pos = np.array([140, 75])
    else:
        goal_pos = np.array([140, 140])

    pos = np.array([inputs[1][0], inputs[1][1]])

    close = np.linalg.norm(pos-goal_pos) <= 10
    out = pos[0] > 150 or pos[0] < 0 or pos[1] > 150 or pos[1] < 0

    if close:
        plus = 100
    elif out:
        plus = -100
    else:
        plus = 0

    theta_ref = normalize_angle(np.arctan2(goal_pos[1]-pos[1], goal_pos[0]-pos[0]))
    theta = inputs[1][2]
    theta = normalize_angle(np.pi/2-theta)
    del_theta = shortest_angular_distance(from_angle=theta, to_angle=theta_ref)
    power = -del_theta**2/(np.pi/6)**2
    res = np.expm1(power)+plus

    return np.array([res])
