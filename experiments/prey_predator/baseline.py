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
    state = ins[0]
    return state[3:5]


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


def build_baseline_agent(alg, params, mdp, horizon, std):
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


def build_computational_graph_baseline(mdp, agent, ep_per_fit,
                                       low_level_callbacks=[]):

    # State Placeholder
    state_ph = PlaceHolder(name='state_ph')

    # Reward Placeholder
    reward_ph = PlaceHolder(name='reward_ph')

    # Last_In Placeholder
    lastaction_ph = PlaceHolder(name='lastaction_ph')

    function_block1 = fBlock(name='compute setpoint',
                             phi=compute_stepoint)

    function_block2 = fBlock(name='angle and distance',
                             phi=polar_error)
    function_block3 = fBlock(name='reward low level', phi=reward_low_level)

    control_block = ControlBlock(name='Control Block', agent=agent,
                                 n_eps_per_fit=ep_per_fit,
                                 callbacks=low_level_callbacks)

    blocks = [state_ph, reward_ph, lastaction_ph,
              function_block1,
              function_block2, function_block3,
              control_block]

    state_ph.add_input(control_block)
    reward_ph.add_input(control_block)
    lastaction_ph.add_input(control_block)

    function_block1.add_input(state_ph)

    function_block2.add_input(state_ph)
    function_block2.add_input(function_block1)

    function_block3.add_input(function_block2)

    control_block.add_input(function_block2)
    control_block.add_reward(function_block3)
    computational_graph = ComputationalGraph(blocks=blocks, model=mdp)

    return computational_graph


def baseline_experiment(mdp, agent,
                        n_epochs, n_episodes, ep_per_eval,
                        ep_per_fit, display, print_j, quiet):
    np.random.seed()

    dataset_callback = CollectDataset()

    computational_graph = build_computational_graph_baseline(
        mdp, agent, ep_per_fit, [dataset_callback])

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
