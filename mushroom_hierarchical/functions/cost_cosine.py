import numpy as np


def cost_cosine(ins):
    del_theta = ins[0]
    reward = np.cos(del_theta)

    return reward
