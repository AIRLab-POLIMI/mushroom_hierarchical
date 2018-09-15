import numpy as np
from mushroom.utils.parameters import Parameter, AdaptiveParameter


class EpsilonUpdate:

    def __init__(self, policy):

        self._policy = policy
        self.counter = 0
        self.last_counter = 0


    def __call__(self, **kwargs):
        dataset = kwargs.get('dataset')
        for step in dataset:
            last = step[-1]
            if last:
                self.counter += 1
        if self.counter % 50 == 0 and not self.last_counter == self.counter:
            new_epsilon = Parameter(self._policy._epsilon.get_value() / 1.01)
            self._policy.set_epsilon(new_epsilon)
            self.last_counter = self.counter

    def get_values(self):

        return self._policy.epsilon

    def reset(self):

        self._policy.epsilon = 0
