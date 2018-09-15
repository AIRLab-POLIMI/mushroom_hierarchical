from mushroom.algorithms.value.td import TD
import numpy as np

from mushroom.utils.eligibility_trace import EligibilityTrace
from mushroom.utils.table import Table


class QLambdaDiscrete(TD):
    """
    Discrete version of SARSA(lambda) algorithm.

    """
    def __init__(self, policy, mdp_info, learning_rate, lambda_coeff,
                 trace='replacing'):
        """
        Constructor.

        Args:
            lambda_coeff (float): eligibility trace coefficient;
            trace (str, 'replacing'): type of eligibility trace to use.

        """
        self.Q = Table(mdp_info.size)
        self._lambda = lambda_coeff

        self.e = EligibilityTrace(self.Q.shape, trace)
        super(QLambdaDiscrete, self).__init__(self.Q, policy, mdp_info,
                                                  learning_rate)

    def _update(self, state, action, reward, next_state, absorbing):
        q_current = self.Q[state, action]

        if not absorbing:
            a_max = np.argmax(self.Q[next_state, :])
            q_next = self.Q[next_state, a_max]
        else:
            q_next = 0.

        delta = reward + self.mdp_info.gamma * q_next - q_current
        self.e.update(state, action)

        self.Q.table += self.alpha(state, action) * delta * self.e.table
        if not absorbing:
            if action == a_max:
                self.e.table *= self.mdp_info.gamma * self._lambda
            else:
                self.e.reset()

    def episode_start(self):
        self.e.reset()

