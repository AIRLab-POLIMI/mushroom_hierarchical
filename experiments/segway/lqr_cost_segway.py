import numpy as np


def lqr_cost_segway(ins):

    x = np.concatenate(ins)
    Q = np.diag([3.0, 0.1, 0.1])
    J = x.dot(Q).dot(x)
    reward = -J

    return np.array([reward])
