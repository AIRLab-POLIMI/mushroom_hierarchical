from mushroom.environments import MDPInfo
from mushroom.features.features import *
from mushroom.features.basis import *
from mushroom.policy.gaussian_policy import *
from mushroom.policy.td_policy import EpsGreedy, Boltzmann
from mushroom.approximators.parametric import LinearApproximator, PyTorchApproximator
from mushroom.approximators.regressor import Regressor
from mushroom.utils.callbacks import CollectDataset
from mushroom.utils.dataset import compute_J, episodes_length
from mushroom.utils import spaces
from mushroom.utils.angles import *


from mushroom_hierarchical.core.hierarchical_core import HierarchicalCore
from mushroom_hierarchical.blocks.computational_graph import ComputationalGraph
from mushroom_hierarchical.blocks.control_block import ControlBlock
from mushroom_hierarchical.blocks.basic_operation_block import *
from mushroom_hierarchical.blocks.model_placeholder import PlaceHolder
from mushroom_hierarchical.blocks.reward_accumulator import *
from mushroom_hierarchical.blocks.hold_state import hold_state

from network import Network


def reward_low_level(ins):
    state = ins[0]

    value = np.cos(state[1]) - state[0]
    return np.array([value])


def compute_stepoint(ins):
    n_actions = 8
    distance = 1.0

    state = ins[0]
    old_state = ins[1]
    action = int(np.asscalar(ins[2]))

    if action == n_actions:
        return state[3:5]
    else:
        theta_target = 2 * np.pi / n_actions * action - np.pi

        del_x = np.cos(theta_target)*distance
        del_y = np.sin(theta_target)*distance

        del_vect = np.array([del_x, del_y])

        return old_state + del_vect


def pick_position(ins):
    state = ins[0]

    return np.concatenate([state[0:2], state[3:5]], 0)


def polar_error(ins):
    state = ins[0]
    ref = ins[1]
    delta_pos = ref - state[:2]

    rho = np.linalg.norm(delta_pos)
    theta_ref = np.arctan2(delta_pos[1], delta_pos[0])


    delta_theta = shortest_angular_distance(from_angle=state[2],
                                            to_angle=theta_ref)

    return np.array([rho, delta_theta])


def build_high_level_agent(alg, params, optim, loss, mdp, horizon_low, eps,
                           n_features, use_cuda):
    high = np.ones(4)
    low = np.zeros(4)

    high[:2] = mdp.info.observation_space.high[:2]
    low[:2] = mdp.info.observation_space.low[:2]

    high[2:] = mdp.info.observation_space.high[3:5]
    low[2:] = mdp.info.observation_space.low[3:5]

    n_actions = 9
    observation_space = spaces.Box(low=low, high=high)
    action_space = spaces.Discrete(n_actions)

    mdp_info = MDPInfo(observation_space=observation_space,
                       action_space=action_space,
                       gamma=mdp.info.gamma**horizon_low,
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


def build_low_level_agent(alg, params, mdp, horizon, std):
    rho_max = np.linalg.norm(mdp.info.observation_space.high[:2] -
                               mdp.info.observation_space.low[:2])
    low = np.array([-np.pi, 0])
    high = np.array([np.pi, rho_max])

    basis = FourierBasis.generate(low, high, 10)
    features = Features(basis_list=basis)

    approximator = Regressor(LinearApproximator,
                             input_shape=(features.size,),
                             output_shape=mdp.info.action_space.shape)

    pi = DiagonalGaussianPolicy(approximator, std)

    mdp_info_agent = MDPInfo(observation_space=spaces.Box(low, high),
                             action_space=mdp.info.action_space,
                             gamma=mdp.info.gamma,
                             horizon=horizon)
    agent = alg(pi, mdp_info_agent, features=features, **params)

    return agent


def build_computational_graph(mdp, agent_low, agent_high,
                              ep_per_fit_low, low_level_callbacks=[]):

    # State Placeholder
    state_ph = PlaceHolder(name='state_ph')

    # Reward Placeholder
    reward_ph = PlaceHolder(name='reward_ph')

    # Last_In Placeholder
    lastaction_ph = PlaceHolder(name='lastaction_ph')

    function_block1 = fBlock(name='pick position',
                             phi=pick_position)

    hold_block = hold_state(name='holdstate')

    function_block2 = fBlock(name='compute setpoint',
                             phi=compute_stepoint)

    function_block3 = fBlock(name='angle and distance',
                             phi=polar_error)
    function_block4 = fBlock(name='reward low level', phi=reward_low_level)

    reward_acc = mean_reward_block(name='mean reward')

    control_block_h = ControlBlock(name='Control Block H', agent=agent_high,
                                   n_steps_per_fit=1)

    control_block_l = ControlBlock(name='Control Block L', agent=agent_low,
                                   n_eps_per_fit=ep_per_fit_low,
                                   callbacks=low_level_callbacks)

    blocks = [state_ph, reward_ph, lastaction_ph,
              control_block_h, control_block_l,
              function_block1, function_block2,
              function_block3, function_block4,
              reward_acc]

    state_ph.add_input(control_block_l)
    reward_ph.add_input(control_block_l)
    lastaction_ph.add_input(control_block_l)

    function_block1.add_input(state_ph)

    reward_acc.add_input(reward_ph)
    reward_acc.add_alarm_connection(control_block_l)

    control_block_h.add_input(function_block1)
    control_block_h.add_reward(reward_acc)
    control_block_h.add_alarm_connection(control_block_l)

    hold_block.add_input(state_ph)
    hold_block.add_alarm_connection(control_block_l)

    function_block2.add_input(state_ph)
    function_block2.add_input(hold_block)
    function_block2.add_input(control_block_h)

    function_block3.add_input(state_ph)
    function_block3.add_input(function_block2)

    function_block4.add_input(function_block3)

    control_block_l.add_input(function_block3)
    control_block_l.add_reward(function_block4)
    computational_graph = ComputationalGraph(blocks=blocks, model=mdp)

    return computational_graph


def experiment(mdp, agent_high, agent_low,
               n_epochs, n_episodes, ep_per_eval,
               ep_per_fit_low, display, print_j, quiet):
    np.random.seed()

    dataset_callback = CollectDataset()

    computational_graph = build_computational_graph(
        mdp, agent_low, agent_high, ep_per_fit_low,
        [dataset_callback])

    core = HierarchicalCore(computational_graph)
    J_list = list()
    L_list = list()

    dataset = core.evaluate(n_episodes=ep_per_eval, quiet=quiet)
    J = compute_J(dataset, gamma=mdp.info.gamma)
    J_list.append(np.mean(J))
    J_low_list = list()
    L = episodes_length(dataset)
    L_list.append(np.mean(L))
    if print_j:
        print('Reward at start :', J_list[-1])

    for n in range(n_epochs):
        core.learn(n_episodes=n_episodes, skip=True, quiet=quiet)

        ll_dataset = dataset_callback.get()
        dataset_callback.clean()
        J_low = compute_J(ll_dataset, mdp.info.gamma)
        J_low_list.append(np.mean(J_low))
        if print_j:
            print('Low level reward at epoch', n, ':', np.mean(J_low))

        dataset = core.evaluate(n_episodes=ep_per_eval, quiet=quiet)
        J = compute_J(dataset, gamma=mdp.info.gamma)
        J_list.append(np.mean(J))
        L = episodes_length(dataset)
        L_list.append(np.mean(L))

        if print_j:
            print('Reward at epoch ', n, ':',  J_list[-1])

        if display:
            core.evaluate(n_episodes=1, render=True)

    return J_list, L_list, J_low_list
