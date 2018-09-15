import numpy as np

from mushroom.core import Core
from mushroom.approximators.regressor import Regressor
from mushroom.approximators.parametric import LinearApproximator
from mushroom.features.features import Features
from mushroom.features.tiles import Tiles
from mushroom.distributions import GaussianDiagonalDistribution
from mushroom.policy import DiagonalGaussianPolicy, DeterministicPolicy
from mushroom.utils.dataset import compute_J, episodes_length


def build_approximator(mdp):
    high = [150, 150, np.pi]
    low = [0, 0, -np.pi]
    n_tiles = [5, 5, 8]
    low = np.array(low, dtype=np.float)
    high = np.array(high, dtype=np.float)
    n_tilings = 1

    tilings = Tiles.generate(n_tilings=n_tilings, n_tiles=n_tiles, low=low,
                             high=high, uniform=True)

    phi = Features(tilings=tilings)

    input_shape = (phi.size,)

    approximator = Regressor(LinearApproximator, input_shape=input_shape,
                             output_shape=mdp.info.action_space.shape)

    return phi, approximator


def build_pg_agent(alg, params, mdp):
    phi, approximator = build_approximator(mdp)

    std = np.array([1e-1])
    policy = DiagonalGaussianPolicy(mu=approximator, std=std)

    agent = alg(policy, mdp.info, features=phi, **params)

    return agent


def build_bbo_agent(alg, params, mdp):
    phi, approximator = build_approximator(mdp)

    policy = DeterministicPolicy(mu=approximator)

    mu = np.zeros(policy.weights_size)
    sigma = 4e-1 * np.ones(policy.weights_size)
    distribution = GaussianDiagonalDistribution(mu, sigma)

    agent = alg(distribution, policy, mdp.info, features=phi, **params)

    return agent


def flat_experiment(mdp, agent, n_epochs, n_iterations,
                    ep_per_iteration, ep_per_eval):
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
        core.learn(n_episodes=n_iterations * ep_per_iteration,
                   n_episodes_per_fit=ep_per_iteration, quiet=True)
        dataset = core.evaluate(n_episodes=ep_per_eval, quiet=True)
        J = compute_J(dataset, gamma=mdp.info.gamma)
        J_list.append(np.mean(J))
        L = episodes_length(dataset)
        L_list.append(np.mean(L))
        #print('J', n, ':', J_list[-1])

    return J_list, L_list