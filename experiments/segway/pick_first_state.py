import numpy as np


def pick_first_state(inputs):
    states = np.concatenate(inputs)
    indices = [0]
    states_needed = np.zeros(len(indices))
    pos = 0
    for i in indices:
        states_needed[pos]=states[i]
        pos += 1

    return states_needed



