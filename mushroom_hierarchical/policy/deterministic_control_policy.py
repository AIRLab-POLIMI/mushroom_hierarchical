import numpy as np


class DeterministicControlPolicy:

    def __init__(self, weights):
        self.__name__ = 'DeterministicPolicy'
        self._weights = weights

    def __call__(self, state, action):
        policy_action = np.atleast_1d(np.abs(self._weights).dot(state))

        return 1. if np.array_equal(action, policy_action) else 0.

    def draw_action(self, state):
        return np.atleast_1d(np.abs(self._weights).dot(state))

    def diff(self, state, action):
        raise RuntimeError('Deterministic policy is not differentiable')

    def diff_log(self, state, action):
        raise RuntimeError('Deterministic policy is not differentiable')

    def set_weights(self, weights):
        self._weights = weights

    def get_weights(self):
        return self._weights

    @property
    def weights_size(self):
        return len(self._weights)

    def __str__(self):
        return self.__name__
