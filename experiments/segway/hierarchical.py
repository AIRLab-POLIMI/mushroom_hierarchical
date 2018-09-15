from mushroom.utils import spaces
from mushroom.environments import MDPInfo
from mushroom.distributions import *
from mushroom.approximators.parametric import LinearApproximator
from mushroom.approximators.regressor import Regressor
from mushroom.utils.dataset import compute_J, episodes_length
from mushroom.policy import DeterministicPolicy

from mushroom_hierarchical.core.hierarchical_core import HierarchicalCore
from mushroom_hierarchical.blocks.computational_graph import ComputationalGraph
from mushroom_hierarchical.blocks.control_block import ControlBlock
from mushroom_hierarchical.blocks.basic_operation_block import *
from mushroom_hierarchical.blocks.model_placeholder import PlaceHolder
from mushroom_hierarchical.utils.callbacks.collect_distribution_parameter import\
    CollectDistributionParameter
from mushroom_hierarchical.policy.deterministic_control_policy import DeterministicControlPolicy

from lqr_cost_segway import lqr_cost_segway
from angle_to_angle_diff_complete_state import *
from pick_first_state import pick_first_state


def fall_reward(inputs):
    state = inputs[0]
    if abs(state[1]) > np.pi / 2:
        res = -10000.0
    else:
        res = 0.0
    return np.array([res])


def lqr_cost_segway(ins):
    x = np.concatenate(ins)
    Q = np.diag([3.0, 0.1, 0.1])
    J = x.dot(Q).dot(x)
    reward = -J

    return np.array([reward])


def build_computational_graph(mdp, agent_low, agent_high, ep_per_fit_low,
                              ep_per_fit_high):
    # State Placeholder
    state_ph = PlaceHolder(name='state_ph')

    # Reward Placeholder
    reward_ph = PlaceHolder(name='reward_ph')

    # Last_In Placeholder
    lastaction_ph = PlaceHolder(name='lastaction_ph')

    # Function Block 1
    function_block1 = fBlock(name='f1 (pick distance to goal state var)',
                             phi=pick_first_state)

    # Function Block 2
    function_block2 = fBlock(name='f2 (build state)',
                             phi=angle_to_angle_diff_complete_state)

    # Function Block 3
    function_block3 = fBlock(name='f3 (reward low level)',
                             phi=lqr_cost_segway)

    # Function Block 4
    function_block4 = addBlock(name='f4 (add block)')

    # Function Block 5
    function_block5 = fBlock(name='f5 (fall punish low level)',
                             phi=fall_reward)

    # Control Block 1
    parameter_callback1 = CollectDistributionParameter(agent_high.distribution)
    control_block_h = ControlBlock(name='Control Block High', agent=agent_high,
                                   n_eps_per_fit=ep_per_fit_high,
                                   callbacks=[parameter_callback1])

    # Control Block 2
    parameter_callback2 = CollectDistributionParameter(agent_low.distribution)
    control_block_l = ControlBlock(name='Control Block Low', agent=agent_low,
                                   n_eps_per_fit=ep_per_fit_low,
                                   callbacks=[parameter_callback2])
    control_block_h.set_mask()

    # Graph
    blocks = [state_ph, reward_ph, lastaction_ph, control_block_h,
              control_block_l, function_block1, function_block2,
              function_block3, function_block4, function_block5]

    state_ph.add_input(control_block_l)
    reward_ph.add_input(control_block_l)
    lastaction_ph.add_input(control_block_l)
    control_block_h.add_input(function_block1)
    control_block_h.add_reward(reward_ph)
    control_block_l.add_input(function_block2)
    control_block_l.add_reward(function_block4)
    function_block1.add_input(state_ph)
    function_block2.add_input(control_block_h)

    function_block2.add_input(state_ph)
    function_block3.add_input(function_block2)
    function_block5.add_input(state_ph)
    function_block4.add_input(function_block3)
    function_block4.add_input(function_block5)
    computational_graph = ComputationalGraph(blocks=blocks, model=mdp)

    return computational_graph, control_block_h


def build_agent_high(alg, params, std, mdp):
    # Features
    approximator1 = Regressor(LinearApproximator,
                             input_shape=(1,),
                             output_shape=(1,))

    # Policy H
    n_weights = approximator1.weights_size
    mu = np.zeros(n_weights)
    sigma = std*np.ones(n_weights)
    pi = DeterministicPolicy(approximator1)
    dist = GaussianDiagonalDistribution(mu, sigma)

    lim = np.pi / 2
    low = mdp.info.observation_space.low[0:1]
    high = mdp.info.observation_space.high[0:1]
    mdp_info = MDPInfo(observation_space=spaces.Box(low, high),
                       action_space=spaces.Box(-lim, lim, (1,)),
                       gamma=mdp.info.gamma,
                       horizon=mdp.info.horizon)
    return alg(dist, pi, mdp_info, **params)


def build_agent_low(alg, params, std, mdp):
    approximator = Regressor(LinearApproximator,
                              input_shape=(3,),
                              output_shape=(1,))
    n_weights = approximator.weights_size
    mu = np.zeros(n_weights)
    sigma = std * np.ones(n_weights)
    pi = DeterministicControlPolicy(approximator)
    dist = GaussianDiagonalDistribution(mu, sigma)

    # Agent Low
    mdp_info = MDPInfo(observation_space=spaces.Box(
        low=mdp.info.observation_space.low[1:],  # FIXME FALSE
        high=mdp.info.observation_space.high[1:],  # FIXME FALSE
        ),
        action_space=mdp.info.action_space,
        gamma=mdp.info.gamma, horizon=mdp.info.horizon)

    return alg(dist, pi, mdp_info, **params)


def hierarchical_experiment(mdp, agent_low, agent_high,
                            n_epochs, n_episodes,
                            ep_per_fit_low, ep_per_fit_high,
                            ep_per_eval):
    np.random.seed()

    computational_graph, control_block_h = build_computational_graph(
        mdp, agent_low, agent_high, ep_per_fit_low, ep_per_fit_high)

    core = HierarchicalCore(computational_graph)
    J_list = list()
    L_list = list()

    dataset = core.evaluate(n_episodes=ep_per_eval, quiet=True)
    J = compute_J(dataset, gamma=mdp.info.gamma)
    J_list.append(np.mean(J))
    L = episodes_length(dataset)
    L_list.append(np.mean(L))

    for n in range(n_epochs):
        if n == 2:
            control_block_h.unset_mask()
        core.learn(n_episodes=n_episodes, skip=True,
                   quiet=True)
        dataset = core.evaluate(n_episodes=ep_per_eval, quiet=True)
        J = compute_J(dataset, gamma=mdp.info.gamma)
        J_list.append(np.mean(J))
        L = episodes_length(dataset)
        L_list.append(np.mean(L))

    return J_list, L_list
