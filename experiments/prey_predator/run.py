import datetime
from joblib import Parallel, delayed


from mushroom.algorithms.policy_search import *
from mushroom.algorithms.value import *
from mushroom.utils.parameters import *

from mushroom.utils.folder import *

from mushroom_hierarchical.environments.prey_predator import PreyPredator
from mushroom_hierarchical.utils.parse_joblib import parse_joblib

import torch.optim as optim
from torch.nn import SmoothL1Loss

from baseline import baseline_experiment, build_baseline_agent
from discretized import discretized_experiment, build_discretized_agent
from hierarchical import *


if __name__ == '__main__':
    n_jobs = -1
    verb = 10

    how_many = 1
    n_epochs = 50
    ep_per_epoch = 1000
    ep_per_eval = 500

    ep_per_fit_low = 10

    n_features = 300
    use_cuda = True
    display = False
    print_j = False
    quiet = True

    mdp = PreyPredator()

    # directory
    name = 'prey_predator'
    subdir = name + '_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')\
             + '/'

    mk_dir_recursive('./' + subdir)
    force_symlink('./' + subdir, name + '_latest')

    # COMMON
    std_low = 1e-1 * np.ones(2)
    horizon = 10

    p_value_gpomdp = 2e-4
    p_gpomdp = dict(
        learning_rate=AdaptiveParameter(value=p_value_gpomdp)
    )

    # BASELINE
    algs_and_params_baseline = [
        (GPOMDP, p_gpomdp)
    ]

    for alg, params in algs_and_params_baseline:
        agent = build_baseline_agent(alg, params, mdp, horizon, std_low)

        print('BASELINE: ', alg.__name__)
        print('lr: ', p_gpomdp['learning_rate'].__class__.__name__,
              p_value_gpomdp)
        res = Parallel(n_jobs=n_jobs, verbose=verb)(
            delayed(baseline_experiment)(mdp, agent,
                                         n_epochs,
                                         ep_per_epoch,
                                         ep_per_eval,
                                         ep_per_fit_low,
                                         display, print_j, quiet)
            for _ in range(how_many))

        J, L, Jlow = parse_joblib(res)
        np.save(subdir + '/J_B_' + alg.__name__, J)
        np.save(subdir + '/L_B_' + alg.__name__, L)
        np.save(subdir + '/Jlow_B_' + alg.__name__, Jlow)

    # DQN discretized
    optimizer_disc = {'class': optim.RMSprop,
                 'params': {'lr': 1e-3,
                            'centered': True}}
    eps_disc = ExponentialDecayParameter(1, -0.2)

    p_dqn_d = dict(
        clip_reward=False,
        initial_replay_size=5000,
        max_replay_size=100000,
        target_update_frequency=200,
        batch_size=500,
        n_approximators=1,
    )
    algs_and_params_discretized = [
        (DQN, p_dqn_d),
        (DoubleDQN, p_dqn_d)
    ]

    n_actions = [4, 4]
    n = 1
    for n_i in n_actions:
        n = n*n_i

    for alg, params in algs_and_params_discretized:
        agent = build_discretized_agent(alg, params, n, optimizer_disc,
                                        SmoothL1Loss(), mdp, eps_disc,
                                        n_features, use_cuda)

        print('DISCRETIZED: ', alg.__name__)
        res = Parallel(n_jobs=n_jobs, verbose=verb)(
            delayed(discretized_experiment)(mdp, agent,
                                            n_actions,
                                            n_epochs,
                                            ep_per_epoch,
                                            ep_per_eval,
                                            display, print_j, quiet)
            for _ in range(how_many))

        J, L = parse_joblib(res)
        np.save(subdir + '/J_D_' + alg.__name__, J)
        np.save(subdir + '/L_D_' + alg.__name__, L)


    # HIERARCHICAL
    optimizer = {'class': optim.RMSprop,
                 'params': {'lr': 1e-3,
                            'centered': True}}

    p_dqn = dict(
        clip_reward=False,
        initial_replay_size=5000,
        max_replay_size=100000,
        target_update_frequency=200,
        batch_size=500,
        n_approximators=1,
    )

    algs_and_params_hier = [
        (DQN, p_dqn, GPOMDP, p_gpomdp),
        (DoubleDQN, p_dqn, GPOMDP, p_gpomdp)
    ]

    eps = ExponentialDecayParameter(1, -0.2)

    for alg_h, params_h, alg_l, params_l in algs_and_params_hier:
        agent_h = build_high_level_agent(alg_h, params_h, optimizer,
                                         SmoothL1Loss(), mdp, horizon, eps,
                                         n_features, use_cuda)
        agent_l = build_low_level_agent(alg_l, params_l, mdp, horizon, std_low)

        print('High: ', alg_h.__name__, ' Low: ', alg_l.__name__)
        print('lr: ', p_gpomdp['learning_rate'].__class__.__name__,
              p_value_gpomdp)
        res = Parallel(n_jobs=n_jobs)(delayed(experiment)
                                      (mdp, agent_h, agent_l,
                                       n_epochs,
                                       ep_per_epoch,
                                       ep_per_eval,
                                       ep_per_fit_low,
                                       display, print_j, quiet)
                                      for _ in range(how_many))

        J, L, Jlow = parse_joblib(res)
        np.save(subdir + '/J_H_' + alg_h.__name__ + '_' + alg_l.__name__, J)
        np.save(subdir + '/L_H_' + alg_h.__name__ + '_' + alg_l.__name__, L)
        np.save(subdir + '/Jlow_H_' + alg_h.__name__ + '_' + alg_l.__name__,
                Jlow)
