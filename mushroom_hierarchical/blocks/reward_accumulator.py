from .block import Block
import numpy as np


class reward_accumulator_block(Block):
    """
    This class implements the block object of a computational graph for hierarchical learning.

    """
    def __init__(self, gamma, name=None):

        self.gamma = gamma
        self.accumulator = None
        self.df = None
        super(reward_accumulator_block, self).__init__(name=name)

    def __call__(self, inputs, reward, absorbing, last, learn_flag, alarms):
        if isinstance(inputs, list):
            inputs = inputs[0]
        self.accumulator += self.df*inputs[0]
        self.df *= self.gamma
        self.last_output = np.array([self.accumulator])

        if np.any(alarms):
            self.accumulator = 0.
            #self.df = 1.

    def reset(self, inputs):
        self.accumulator = 0.
        self.df = 1.
        self.last_output = np.zeros(1)

    def init(self):
        self.last_output = None


class mean_reward_block(Block):
    """
    This class implements the block object of a computational graph for hierarchical learning.

    """
    def __init__(self, name=None):
        self.accumulator = None
        self.n = None
        super(mean_reward_block, self).__init__(name=name)

    def __call__(self, inputs, reward, absorbing, last, learn_flag, alarms):
        if isinstance(inputs, list):
            inputs = inputs[0]
        self.accumulator += (inputs[0] - self.accumulator)/self.n
        self.n += 1
        self.last_output = np.array([self.accumulator])

        if np.any(alarms):
            self.accumulator = 0.
            self.n = 1.

    def reset(self, inputs):
        self.accumulator = 0.
        self.n = 1.
        self.last_output = np.zeros(1)

    def init(self):
        self.last_output = None

