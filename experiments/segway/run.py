import datetime
from joblib import Parallel, delayed

from mushroom.algorithms.policy_search import *

from mushroom.utils.folder import *

from mushroom_hierarchical.utils.parse_joblib import parse_joblib
from mushroom_hierarchical.environments.segway_linear_motion import *

from flat import *
from hierarchical import *


if __name__ == '__main__':
    n_jobs = -1

    how_many = 100
    n_epochs = 25
    n_episodes = 100
    ep_per_eval = 100

    ep_per_fit_bbo = 25
    ep_per_fit_low = 25
    ep_per_fit_high = 50

    mdp = SegwayLinearMotion(goal_distance=1.0)

    # directory
    name = 'segway'
    subdir = name + '_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')\
             + '/'

    mk_dir_recursive('./' + subdir)
    force_symlink('./' + subdir, name + '_latest')

    # FLAT BBO
    std_bbo = 2e-0
    algs_and_params_bbo = [
        (REPS, {'eps': 5e-2}),
        (RWR, {'beta': 2e-3})
    ]

    for alg, params in algs_and_params_bbo:
        agent = build_bbo_agent(alg, params, std_bbo, mdp)

        print(alg.__name__)
        res = Parallel(n_jobs=n_jobs)(delayed(flat_experiment)(mdp, agent,
                                                               n_epochs,
                                                               n_episodes,
                                                               ep_per_fit_bbo,
                                                               ep_per_eval)
                                      for _ in range(how_many))
        J, L = parse_joblib(res)
        np.save(subdir + '/J_' + alg.__name__, J)
        np.save(subdir + '/L_' + alg.__name__, L)

    # HIERARCHICAL
    algs_and_params_hier = [
        (RWR, {'beta': 1e-2}, RWR, {'beta': 2e-3}),
        (RWR, {'beta': 1e-2}, REPS, {'eps': 5e-2}),
        (REPS, {'eps': 5e-2}, RWR, {'beta': 2e-3})
    ]

    std_high = 2.0e-2
    std_low = 2.0

    for alg_h, params_h, alg_l, params_l in algs_and_params_hier:
        agent_high = build_agent_high(alg_h, params_h, std_high, mdp)
        agent_low = build_agent_low(alg_l, params_l, std_low, mdp)

        print('High: ', alg_h.__name__, ' Low: ', alg_l.__name__)
        res = Parallel(n_jobs=n_jobs)(delayed(hierarchical_experiment)
                                      (mdp, agent_low, agent_high,
                                       n_epochs, n_episodes,
                                       ep_per_fit_low, ep_per_fit_high,
                                       ep_per_eval)
                                      for _ in range(how_many))

        J, L = parse_joblib(res)
        np.save(subdir + '/J_H_' + alg_h.__name__ + '_' + alg_l.__name__, J)
        np.save(subdir + '/L_H_' + alg_h.__name__ + '_' + alg_l.__name__, L)
