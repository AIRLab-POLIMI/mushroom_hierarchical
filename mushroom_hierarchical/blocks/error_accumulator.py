from .block import Block
import numpy as np


class ErrorAccumulatorBlock(Block):
 
    def __init__(self, name=None):

        self.accumulator = None
        super(ErrorAccumulatorBlock, self).__init__(name=name)

    def __call__(self, inputs, reward, absorbing, last, learn_flag, alarms):

        if last or np.any(alarms):
            self.accumulator = 0
        else:
            if isinstance(inputs, list):
                inputs = inputs[0]
            self.accumulator += inputs[0]

        self.last_output = np.array([self.accumulator])


    def reset(self, inputs):
        self.accumulator = 0
        self.last_output = np.zeros(1)

    def init(self):
        self.last_output = None

