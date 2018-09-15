from .block import Block
import numpy as np


class fBlockAlarm(Block):

    def __init__(self, phi, name=None):
        self.phi = phi
        super(fBlockAlarm, self).__init__(name=name)

    def _call(self, inputs, reward, absorbing, last, learn_flag):
        self.last_output = None
        self.alarm_output = self.phi(inputs)

    def reset(self, inputs):
        self.last_output = None
        self.alarm_output = self.phi(inputs)

    def init(self):
        self.last_output = None
        self.alarm_output = None
