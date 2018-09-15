import datetime
from joblib import Parallel, delayed

from mushroom.environments import ShipSteering
from mushroom.algorithms.policy_search import *
from mushroom.utils.parameters import AdaptiveParameter

from mushroom.utils.folder import *

from mushroom_hierarchical.utils.parse_joblib import parse_joblib

from flat import *
from hierarchical import *


if __name__ == '__main__':
    n_jobs = -1

    how_many = 100
    n_epochs = 25
    ep_per_epoch = 200
    ep_per_eval = 50

    n_iterations_hier = 10
    n_iterations_bbo = 10
    n_iterations_pg = 5

    ep_per_run_low = 10

    mdp = ShipSteering(small=True, n_steps_action=3)

    # directory
    name = 'ship_steering_small'
    subdir = name + '_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')\
             + '/'

    mk_dir_recursive('./' + subdir)
    force_symlink('./' + subdir, name + '_latest')

    # FLAT PG
    '''algs_and_params_pg = [
        (GPOMDP, {'learning_rate': AdaptiveParameter(value=1e-5)})
    ]
    for alg, params in algs_and_params_pg:
        agent = build_pg_agent(alg, params, mdp)
        ep_per_run_pg = ep_per_epoch // n_iterations_pg

        print(alg.__name__)
        res = Parallel(n_jobs=n_jobs)(delayed(flat_experiment)(mdp,
                                                               agent,
                                                               n_epochs,
                                                               n_iterations_pg,
                                                               ep_per_run_pg,
                                                               ep_per_eval)
                                      for _ in range(how_many))

        J, L = parse_joblib(res)
        np.save(subdir + '/J_' + alg.__name__, J)
        np.save(subdir + '/L_' + alg.__name__, L)

    # FLAT BBO
    algs_and_params_bbo = [
        (REPS, {'eps': 1.0}),
        (RWR, {'beta': 0.7}),
        (PGPE, {'learning_rate': AdaptiveParameter(value=1.5)}),
    ]

    for alg, params in algs_and_params_bbo:
        agent = build_bbo_agent(alg, params, mdp)

        ep_per_run_bbo = ep_per_epoch // n_iterations_bbo

        print(alg.__name__)
        res = Parallel(n_jobs=n_jobs)(delayed(flat_experiment)(mdp,
                                                               agent,
                                                               n_epochs,
                                                               n_iterations_bbo,
                                                               ep_per_run_bbo,
                                                               ep_per_eval)
                                      for _ in range(how_many))
        J, L = parse_joblib(res)
        np.save(subdir + '/J_' + alg.__name__, J)
        np.save(subdir + '/L_' + alg.__name__, L)'''

    # HIERARCHICAL
    algs_and_params_hier = [
        (GPOMDP, {'learning_rate': AdaptiveParameter(value=10)},
         PGPE, {'learning_rate': AdaptiveParameter(value=5e-4)})
    ]

    mu = np.array([75, 75])
    sigma = np.array([40, 40])

    for alg_h, params_h, alg_l, params_l in algs_and_params_hier:
        agent_h = build_high_level_agent(alg_h, params_h, mdp, mu, sigma)
        agent_l = build_low_level_agent(alg_l, params_l, mdp)

        ep_per_run_hier = ep_per_epoch // n_iterations_hier

        print('High: ', alg_h.__name__, ' Low: ', alg_l.__name__)
        res = Parallel(n_jobs=n_jobs)(delayed(hierarchical_experiment)
                                      (mdp, agent_l, agent_h,
                                       n_epochs, n_iterations_hier,
                                       ep_per_run_hier, ep_per_eval,
                                       ep_per_run_low)
                                      for _ in range(how_many))

        J, L = parse_joblib(res)
        np.save(subdir + '/J_H_' + alg_h.__name__ + '_' + alg_l.__name__, J)
        np.save(subdir + '/L_H_' + alg_h.__name__ + '_' + alg_l.__name__, L)
