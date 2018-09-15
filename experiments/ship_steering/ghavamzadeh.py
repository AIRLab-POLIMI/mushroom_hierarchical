from mushroom.environments import MDPInfo
from mushroom.features.tiles import Tiles
from mushroom.features.features import *
from mushroom.policy.gaussian_policy import *
from mushroom.policy.td_policy import EpsGreedy
from mushroom.approximators.parametric import LinearApproximator
from mushroom.approximators.regressor import Regressor
from mushroom.utils.parameters import Parameter
from mushroom.utils.dataset import compute_J, episodes_length
from mushroom.utils import spaces

from mushroom_hierarchical.core.hierarchical_core import HierarchicalCore
from mushroom_hierarchical.utils.callbacks.epsilon_update import EpsilonUpdate
from mushroom_hierarchical.blocks.computational_graph import ComputationalGraph
from mushroom_hierarchical.blocks.control_block import ControlBlock
from mushroom_hierarchical.blocks.reward_accumulator import \
    reward_accumulator_block
from mushroom_hierarchical.blocks.basic_operation_block import *
from mushroom_hierarchical.blocks.model_placeholder import PlaceHolder
from mushroom_hierarchical.blocks.mux_block import MuxBlock
from mushroom_hierarchical.blocks.hold_state import hold_state
from mushroom_hierarchical.blocks.discretization_block import \
    DiscretizationBlock

from rototranslate import rototranslate
from ext_reward import *


class TerminationCondition(object):

    def __init__(self, active_dir):
        self.active_direction = active_dir

    def __call__(self, state):
        if self.active_direction == '+':
            goal_pos = np.array([140, 75])
        elif self.active_direction == 'x':
            goal_pos = np.array([140, 140])

        pos = np.array([state[0], state[1]])
        if np.linalg.norm(pos-goal_pos) <= 10 \
                or pos[0] > 150 or pos[0] < 0 \
                or pos[1] > 150 or pos[1] < 0:
            #if np.linalg.norm(pos-goal_pos) <= 10:
            #    print('reached ', self.active_direction)
            return True
        else:
            return False


def selector_function(inputs):
    action = np.asscalar(inputs[0])
    return 0 if action < 4 else 1


def pick_state(inputs):
    states = np.concatenate(inputs)
    indices = [0,1]
    states_needed = np.zeros(len(indices))
    pos = 0
    for i in indices:
        states_needed[pos]=states[i]
        pos += 1

    return states_needed


def build_high_level_ghavamzadeh(alg, params, mdp):
    epsilon = Parameter(value=0.1)
    pi = EpsGreedy(epsilon=epsilon)
    gamma = 1.0
    mdp_info_agentH = MDPInfo(
        observation_space=spaces.Discrete(400),
        action_space=spaces.Discrete(8), gamma=gamma, horizon=10000)

    agent = alg(policy=pi,
                mdp_info=mdp_info_agentH,
                **params)

    return agent


def build_low_level_ghavamzadeh(alg, params, mdp):
    # FeaturesL
    high = [150, 150, np.pi]
    low = [0, 0, -np.pi]
    n_tiles = [5, 5, 10]
    low = np.array(low, dtype=np.float)
    high = np.array(high, dtype=np.float)
    n_tilings = 3

    tilingsL = Tiles.generate(n_tilings=n_tilings, n_tiles=n_tiles,
                              low=low, high=high)

    featuresL = Features(tilings=tilingsL)

    mdp_info_agentL = MDPInfo(
        observation_space=spaces.Box(low=np.array([0, 0]),
                                     high=np.array([150, 150]),
                                     shape=(2,)),
        action_space=mdp.info.action_space, gamma=0.99, horizon=10000)

    input_shape = (featuresL.size,)
    approximator = Regressor(LinearApproximator, input_shape=input_shape,
                              output_shape=mdp.info.action_space.shape)

    std = np.array([3e-2])
    pi = DiagonalGaussianPolicy(mu=approximator, std=std)

    agent = alg(pi, mdp_info_agentL, features=featuresL, **params)

    return agent


def build_ghavamzadeh_graph(mdp, agent_plus, agent_cross, agent_high,
                              ep_per_fit_low):
    # State Placeholder
    state_ph = PlaceHolder(name='state_ph')

    # Reward Placeholder
    reward_ph = PlaceHolder(name='reward_ph')

    # Last action Placeholder
    lastaction_ph = PlaceHolder(name='lastaction_ph')

    # FeaturesH
    low_hi = 0
    lim_hi = 1000 + 1e-8
    n_tiles_high = [20, 20]

    # Discretization Block
    discretization_block = DiscretizationBlock(low=low_hi, high=lim_hi,
                                               n_tiles=n_tiles_high)

    # Control Block H
    control_blockH = ControlBlock(name='control block H',
                                  agent=agent_high,
                                  n_steps_per_fit=1)

    # Termination Conds
    termination_condition1 = TerminationCondition(active_dir='+')
    termination_condition2 = TerminationCondition(active_dir='x')

    # Control Block +
    control_block_plus = ControlBlock(
        name='control block 1', agent=agent_plus,
        n_eps_per_fit=ep_per_fit_low,
        termination_condition=termination_condition1)

    # Control Block x
    control_block_cross = ControlBlock(
        name='control block 2', agent=agent_cross,
        n_eps_per_fit=ep_per_fit_low,
        termination_condition=termination_condition2)

    # Function Block 1: picks state for hi lev ctrl
    function_block1 = fBlock(phi=pick_state, name='f1 pickstate')

    # Function Block 2: maps the env to low lev ctrl state
    function_block2 = fBlock(phi=rototranslate, name='f2 rotot')

    # Function Block 3: holds curr state as ref
    function_block3 = hold_state(name='f3 holdstate')

    # Function Block 4: adds hi lev rew
    function_block4 = addBlock(name='f4 add')

    # Function Block 5: adds low lev rew
    function_block5 = addBlock(name='f5 add')

    # Function Block 6:ext rew of hi lev ctrl
    function_block6 = fBlock(phi=G_high, name='f6 G_hi')

    # Function Block 7: ext rew of low lev ctrl
    function_block7 = fBlock(phi=G_low, name='f7 G_lo')

    # Reward Accumulator H:
    reward_acc_H = reward_accumulator_block(gamma=1.0,
                                            name='reward_acc_H')

    # Selector Block
    function_block8 = fBlock(phi=selector_function, name='f7 G_lo')

    # Mux_Block
    mux_block = MuxBlock(name='mux')
    mux_block.add_block_list([control_block_plus])
    mux_block.add_block_list([control_block_cross])

    # Algorithm
    blocks = [state_ph, reward_ph, lastaction_ph, control_blockH, mux_block,
              function_block1, function_block2, function_block3,
              function_block4, function_block5,
              function_block6, function_block7, function_block8,
              reward_acc_H, discretization_block]

    reward_acc_H.add_input(reward_ph)
    reward_acc_H.add_alarm_connection(control_block_plus)
    reward_acc_H.add_alarm_connection(control_block_cross)

    control_blockH.add_input(discretization_block)
    control_blockH.add_reward(function_block4)
    control_blockH.add_alarm_connection(control_block_plus)
    control_blockH.add_alarm_connection(control_block_cross)

    mux_block.add_input(function_block8)
    mux_block.add_input(function_block2)

    control_block_plus.add_reward(function_block5)
    control_block_cross.add_reward(function_block5)

    function_block1.add_input(state_ph)

    function_block2.add_input(control_blockH)
    function_block2.add_input(state_ph)
    function_block2.add_input(function_block3)

    function_block3.add_input(state_ph)
    function_block3.add_alarm_connection(control_block_plus)
    function_block3.add_alarm_connection(control_block_cross)

    function_block4.add_input(function_block6)
    function_block4.add_input(reward_acc_H)

    function_block5.add_input(function_block7)

    function_block6.add_input(reward_ph)

    function_block7.add_input(control_blockH)
    function_block7.add_input(function_block2)

    function_block8.add_input(control_blockH)

    discretization_block.add_input(function_block1)

    computational_graph = ComputationalGraph(blocks=blocks, model=mdp)

    return computational_graph, control_blockH


def ghavamzadeh_experiment(mdp, agent_plus, agent_cross, agent_high,
                            n_epochs, n_episodes, ep_per_eval,
                            ep_per_iteration_low):
    np.random.seed()

    computational_graph, control_blockH = build_ghavamzadeh_graph(
        mdp, agent_plus, agent_cross, agent_high,
        ep_per_iteration_low)

    core = HierarchicalCore(computational_graph)
    J_list = list()
    L_list = list()

    epsilon_update = EpsilonUpdate(agent_high.policy)

    dataset = core.evaluate(n_episodes=ep_per_eval, quiet=True)
    J = compute_J(dataset, gamma=mdp.info.gamma)
    J_list.append(np.mean(J))
    L = episodes_length(dataset)
    L_list.append(np.mean(L))

    for n in range(n_epochs):
        core.learn(n_episodes=n_episodes, skip=True, quiet=True)
        dataset = core.evaluate(n_episodes=ep_per_eval, quiet=True)
        J = compute_J(dataset, gamma=mdp.info.gamma)
        J_list.append(np.mean(J))
        L = episodes_length(dataset)
        L_list.append(np.mean(L))

        if n == 4:
            control_blockH.callbacks = [epsilon_update]

    return J_list, L_list
