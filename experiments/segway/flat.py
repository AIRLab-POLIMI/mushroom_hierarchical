import numpy as np

from mushroom.core import Core
from mushroom.distributions import GaussianDiagonalDistribution
from mushroom.utils.dataset import compute_J, episodes_length
from mushroom.utils.angles import shortest_angular_distance

class SegwayControlPolicy:
    def __init__(self, weights):
        self._weights = weights

    def __call__(self, state, action):
        policy_action = np.atleast_1d(np.abs(self._weights).dot(state))

        return 1. if np.array_equal(action, policy_action) else 0.

    def draw_action(self, state):
        angle_setpoint = state[0]*self._weights[0]

        new_state = state[1:]

        new_state[0] = shortest_angular_distance(angle_setpoint, new_state[0])

        return np.atleast_1d(np.abs(self._weights[1:]).dot(new_state))

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


def build_bbo_agent(alg, params, std, mdp):
    input_dim = mdp.info.observation_space.shape[0]
    mu = np.zeros(input_dim)
    sigma = std * np.ones(input_dim)
    policy = SegwayControlPolicy(mu)
    dist = GaussianDiagonalDistribution(mu, sigma)
    agent = alg(dist, policy, mdp.info, **params)

    return agent


def flat_experiment(mdp, agent, n_epochs, n_episodes,
                    ep_per_fit, ep_per_eval):
    np.random.seed()

    J_list = list()
    L_list = list()
    core = Core(agent, mdp)

    dataset = core.evaluate(n_episodes=ep_per_eval, quiet=True)
    J = compute_J(dataset, gamma=mdp.info.gamma)
    J_list.append(np.mean(J))
    L = episodes_length(dataset)
    L_list.append(np.mean(L))

    for n in range(n_epochs):
        core.learn(n_episodes=n_episodes,
                   n_episodes_per_fit=ep_per_fit, quiet=True)
        dataset = core.evaluate(n_episodes=ep_per_eval, quiet=True)
        J = compute_J(dataset, gamma=mdp.info.gamma)
        J_list.append(np.mean(J))
        L = episodes_length(dataset)
        L_list.append(np.mean(L))

    return J_list, L_list