from mushroom.environments import MDPInfo
from mushroom.policy.td_policy import Boltzmann
from mushroom.approximators.parametric import PyTorchApproximator

from mushroom.utils.dataset import compute_J, episodes_length
from mushroom.utils import spaces
from mushroom.utils.angles import *

from mushroom_hierarchical.core.hierarchical_core import HierarchicalCore
from mushroom_hierarchical.blocks.computational_graph import ComputationalGraph
from mushroom_hierarchical.blocks.control_block import ControlBlock
from mushroom_hierarchical.blocks.basic_operation_block import *
from mushroom_hierarchical.blocks.model_placeholder import PlaceHolder
from mushroom_hierarchical.blocks.reward_accumulator import *

import itertools

from network import Network


class ActionConverter:
    def __init__(self, n_actions, mdp_info):
        low = mdp_info.action_space.low
        high = mdp_info.action_space.high

        value_list = list()

        for n, l, h in zip(n_actions, low, high):
            value_list.append(list())

            assert(n > 0)

            if (n == 1):
                value_list[-1].append(l+h/2)
            else:
                step = (h - l)/n
                value_list[-1].append(l)

                for i in range(n-2):
                    v = l + (i+1)*step
                    value_list[-1].append(v)

                value_list[-1].append(h)

        actions = itertools.product(*value_list)
        self._actions = [np.array(x) for x in actions]

        print('Actions used:')
        for i, a in enumerate(self._actions):
            print(i, ':', a)

    def __call__(self, ins):
        i = int(np.asscalar(ins[0]))
        return self._actions[i]


def build_discretized_agent(alg, params, n, optim, loss, mdp, eps,
                            n_features, use_cuda):
    high = mdp.info.observation_space.high
    low = mdp.info.observation_space.low

    observation_space = spaces.Box(low=low, high=high)
    action_space = spaces.Discrete(n)

    mdp_info = MDPInfo(observation_space=observation_space,
                       action_space=action_space,
                       gamma=mdp.info.gamma,
                       horizon=mdp.info.horizon)

    pi = Boltzmann(eps)

    approximator_params = dict(network=Network,
                               optimizer=optim,
                               loss=loss,
                               n_features=n_features,
                               input_shape=mdp_info.observation_space.shape,
                               output_shape=mdp_info.action_space.size,
                               n_actions=mdp_info.action_space.n,
                               use_cuda=use_cuda)

    agent = alg(PyTorchApproximator, pi, mdp_info,
                approximator_params=approximator_params, **params)

    return agent


def build_computational_graph_discretized(mdp, agent, n_actions):

    # State Placeholder
    state_ph = PlaceHolder(name='state_ph')

    # Reward Placeholder
    reward_ph = PlaceHolder(name='reward_ph')

    # Last_In Placeholder
    lastaction_ph = PlaceHolder(name='lastaction_ph')

    control_block = ControlBlock(name='Control Block H', agent=agent,
                                 n_steps_per_fit=1)

    function_block = fBlock(name='Action converter',
                            phi=ActionConverter(n_actions, mdp.info))

    blocks = [state_ph, reward_ph, lastaction_ph,
              control_block, function_block]

    state_ph.add_input(function_block)
    reward_ph.add_input(function_block)
    lastaction_ph.add_input(function_block)

    control_block.add_input(state_ph)
    control_block.add_reward(reward_ph)

    function_block.add_input(control_block)

    computational_graph = ComputationalGraph(blocks=blocks, model=mdp)

    return computational_graph


def discretized_experiment(mdp, agent, n_actions, n_epochs, n_episodes,
                           ep_per_eval, display, print_j, quiet):
    np.random.seed()

    computational_graph = build_computational_graph_discretized(
        mdp, agent, n_actions)

    core = HierarchicalCore(computational_graph)
    J_list = list()
    L_list = list()

    dataset = core.evaluate(n_episodes=ep_per_eval, quiet=quiet)
    J = compute_J(dataset, gamma=mdp.info.gamma)
    J_list.append(np.mean(J))
    L = episodes_length(dataset)
    L_list.append(L)
    if print_j:
        print('Reward at start :', J_list[-1])

    for n in range(n_epochs):
        core.learn(n_episodes=n_episodes, skip=True, quiet=quiet)
        dataset = core.evaluate(n_episodes=ep_per_eval, quiet=quiet)
        J = compute_J(dataset, gamma=mdp.info.gamma)
        J_list.append(np.mean(J))
        L = episodes_length(dataset)
        L_list.append(np.mean(L))

        if print_j:
            print('Reward at epoch ', n, ':',  J_list[-1])

        if display:
            core.evaluate(n_episodes=1, render=True)

    return J_list, L_list
