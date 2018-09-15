from .block import Block
import numpy as np


class hold_state(Block):

    def __init__(self, name=None):

        super(hold_state, self).__init__(name=name)

    def __call__(self, inputs, reward, absorbing, last, learn_flag, alarms):

        if np.any(alarms):
            states = np.concatenate(inputs)
            states = np.array([states[0], states[1]])
            self.last_states = states
            self.last_output = self.last_states
        else:
            states = np.concatenate(inputs)
            self.last_states = np.array([states[0], states[1]])

    def reset(self, inputs):
        states = np.concatenate(inputs)
        self.last_output = np.array([states[0], states[1]])

    def init(self):
        self.last_output = None
