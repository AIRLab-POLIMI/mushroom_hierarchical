from mushroom.policy import EpsGreedy
from mushroom.environments import MDPInfo
from mushroom.approximators.parametric import LinearApproximator
from mushroom.approximators.regressor import Regressor
from mushroom.distributions import GaussianDiagonalDistribution
from mushroom.policy.gaussian_policy import *
from mushroom.utils.dataset import compute_J
from mushroom.utils import spaces
from mushroom.features.features import *
from mushroom.features.basis import PolynomialBasis
from mushroom.features.tiles import Tiles
from mushroom.utils.parameters import AdaptiveParameter, Parameter


from mushroom_hierarchical.core.hierarchical_core import HierarchicalCore
from mushroom_hierarchical.blocks.computational_graph import ComputationalGraph
from mushroom_hierarchical.blocks.control_block import ControlBlock
from mushroom_hierarchical.blocks.mux_block import MuxBlock
from mushroom_hierarchical.functions.feature_angle_diff_ship_steering\
    import *
from mushroom_hierarchical.blocks.basic_operation_block import *
from mushroom_hierarchical.blocks.model_placeholder import PlaceHolder
from mushroom_hierarchical.blocks.reward_accumulator import *
from mushroom_hierarchical.functions.cost_cosine import cost_cosine
from mushroom_hierarchical.policy.deterministic_control_policy \
    import DeterministicControlPolicy


def count_gates(dataset):
    gates = list()

    for i in range(len(dataset)):
        if dataset[i][-1]:
            gates.append(np.sum(dataset[i][0][4:]))

    return np.mean(gates)


def hi_lev_state(ins):
    state = np.concatenate(ins)
    #res = 0

    #for i in [4, 5, 6, 7]:
    #    if state[i] > 0:
    #        res += 2**(i-4)

    return np.array([state[0], state[1]])


def compute_pos_ref(ins):
    theta_ref = ins[0]
    state = ins[1]
    x = state[0]
    y = state[1]
    x_ref = x + 10*np.cos(theta_ref)
    y_ref = y + 10*np.sin(theta_ref)

    return np.array([x_ref, y_ref])


class TerminationConditionLow(object):

    def __init__(self, small):
        self.small = small

    def __call__(self, state):

        if self.small:
            lim = 0.25
        else:
            lim = 1
        pos_diff = state[1]
        if pos_diff <= lim:
            return True
        else:
            return False


def build_high_level_agent(alg, params, mdp, mu, std):
    tilings = Tiles.generate(n_tilings=1, n_tiles=[10, 10], low=mdp.info.observation_space.low[:2], high=mdp.info.observation_space.high[:2])
    features = Features(tilings=tilings)

    input_shape = (features.size,)


    mu_approximator = Regressor(LinearApproximator, input_shape=input_shape,
                                output_shape=(1,))
    std_approximator = Regressor(LinearApproximator, input_shape=input_shape,
                                 output_shape=(1,))

    w_mu = mu*np.ones(mu_approximator.weights_size)
    mu_approximator.set_weights(w_mu)

    w_std = std * np.ones(std_approximator.weights_size)
    mu_approximator.set_weights(w_std)

    pi = StateLogStdGaussianPolicy(mu=mu_approximator,
                                log_std=std_approximator)


    obs_low = np.array([mdp.info.observation_space.low[0], mdp.info.observation_space.low[1]])
    obs_high = np.array([mdp.info.observation_space.high[0], mdp.info.observation_space.high[1]])
    mdp_info_agent1 = MDPInfo(observation_space=spaces.Box(obs_low,
                                                           obs_high,
                                                           shape=(2,)),
                              action_space=spaces.Box(mdp.info.observation_space.low[2],
                                                      mdp.info.observation_space.high[2],
                                                      shape=(1,)),
                              gamma=1,
                              horizon=10)
    agent = alg(policy=pi, mdp_info=mdp_info_agent1, features=features, **params)

    return agent


def build_low_level_agent(alg, params, mdp):
    features = Features(basis_list=[PolynomialBasis(dimensions=[0], degrees=[1])])

    pi = DeterministicControlPolicy(weights=np.array([0]))
    mu = np.zeros(pi.weights_size)
    sigma = 1e-3 * np.ones(pi.weights_size)
    distribution = GaussianDiagonalDistribution(mu, sigma)

    mdp_info_agent2 = MDPInfo(observation_space=spaces.Box(np.array([-np.pi, -500]), np.array([np.pi, 500]), (2,)),
                              action_space=mdp.info.action_space,
                              gamma=mdp.info.gamma, horizon=100)
    agent = alg(distribution, pi, mdp_info_agent2, features=features, **params)

    return agent


def build_computational_graph(mdp, agent_low, agent_high,
                              ep_per_fit_low, ep_per_fit_high):

    # State Placeholder
    state_ph = PlaceHolder(name='state_ph')

    # Reward Placeholder
    reward_ph = PlaceHolder(name='reward_ph')

    # Last_In Placeholder
    lastaction_ph = PlaceHolder(name='lastaction_ph')

    # Function Block 0
    function_block0 = fBlock(name='f0 (state build for high level)', phi=hi_lev_state)

    # Function Block 1
    function_block1 = fBlock(name='f1 (angle difference)',
                             phi=angle_ref_angle_difference)

    # Function Block 2
    function_block2 = fBlock(name='f2 (cost cosine)', phi=cost_cosine)

    # Function Block 3
    function_block3 = fBlock(name='f2 (compute pos ref)', phi=compute_pos_ref)

    # Cotrol Block H
    control_block_h = ControlBlock(name='Control Block H', agent=agent_high,
                                    n_eps_per_fit=ep_per_fit_high)
    # Control Block L
    term_cond_low = TerminationConditionLow(small=mdp.small)
    control_block_l = ControlBlock(name='Control Block L', agent=agent_low,
                                   n_eps_per_fit=ep_per_fit_low, termination_condition=term_cond_low)

    # Reward Accumulators
    reward_acc = reward_accumulator_block(gamma=mdp.info.gamma,
                                             name='reward_acc')


    # Algorithm
    blocks = [state_ph, reward_ph, lastaction_ph, control_block_h, reward_acc,
              control_block_l, function_block0, function_block1, function_block2, function_block3]

    state_ph.add_input(control_block_l)
    reward_ph.add_input(control_block_l)
    lastaction_ph.add_input(control_block_l)

    control_block_h.add_input(function_block0)
    control_block_h.add_reward(reward_acc)
    control_block_h.add_alarm_connection(control_block_l)


    reward_acc.add_input(reward_ph)
    reward_acc.add_alarm_connection(control_block_l)

    function_block0.add_input(state_ph)

    function_block1.add_input(control_block_h)
    function_block1.add_input(function_block3)
    function_block1.add_input(state_ph)

    function_block2.add_input(function_block1)

    function_block3.add_input(control_block_h)
    function_block3.add_input(state_ph)
    function_block3.add_alarm_connection(control_block_l)

    control_block_l.add_input(function_block1)
    control_block_l.add_reward(function_block2)

    computational_graph = ComputationalGraph(blocks=blocks, model=mdp)

    return computational_graph


def two_level_hierarchical_experiment(mdp, agent_l, agent_h, n_epochs,
                            n_iterations, ep_per_epoch_train,
                            ep_per_epoch_eval, ep_per_fit_low, ep_per_fit_high):
    np.random.seed()

    computational_graph = build_computational_graph(mdp, agent_l,
                                                    agent_h,
                                                    ep_per_fit_low, ep_per_fit_high)

    core = HierarchicalCore(computational_graph)
    J_list = list()
    dataset = core.evaluate(n_episodes=ep_per_epoch_eval, quiet=True)
    J = compute_J(dataset, gamma=mdp.info.gamma)
    J_list.append(np.mean(J))
    print('J at start: ', np.mean(J))
    print('Mean gates passed: ', count_gates(dataset))

    for n in range(n_epochs):

        core.learn(n_episodes=n_iterations * ep_per_epoch_train, skip=True,
                   quiet=False)
        dataset = core.evaluate(n_episodes=ep_per_epoch_eval, quiet=True, render=True)
        J = compute_J(dataset, gamma=mdp.info.gamma)
        J_list.append(np.mean(J))
        print('J at iteration ', n, ': ', np.mean(J))
        print('Mean gates passed: ', count_gates(dataset))



    return J_list

