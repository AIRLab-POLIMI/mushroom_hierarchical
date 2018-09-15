import datetime
from joblib import Parallel, delayed

from mushroom.environments import ShipSteering
from mushroom.algorithms.policy_search import *
from mushroom.utils.parameters import AdaptiveParameter

from mushroom.utils.folder import *

from mushroom_hierarchical.agents.Q_lambda_discrete import QLambdaDiscrete
from mushroom_hierarchical.utils.parse_joblib import parse_joblib

from hierarchical import *
from ghavamzadeh import *


if __name__ == '__main__':
    n_jobs = -1

    how_many = 100
    n_epochs = 50
    ep_per_epoch = 800
    ep_per_eval = 50

    n_iterations_hier = 10

    ep_per_run_low = 10
    ep_per_run_low_ghavamzadeh = 50


    mdp = ShipSteering(small=False, n_steps_action=3)

    # directory
    name = 'ship_steering_big'
    subdir = name + '_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')\
             + '/'

    mk_dir_recursive('./' + subdir)
    force_symlink('./' + subdir, name + '_latest')

    # HIERARCHICAL
    '''algs_and_params_hier = [
        (GPOMDP, {'learning_rate': AdaptiveParameter(value=50)},
         PGPE, {'learning_rate': AdaptiveParameter(value=5e-4)})
    ]

    mu = np.array([500, 500])
    sigma = np.array([255, 255])

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
        np.save(subdir + '/L_H_' + alg_h.__name__ + '_' + alg_l.__name__, L)'''

    # GHAVAMZADEH
    params_high = {'learning_rate': Parameter(value=8e-2), 'lambda_coeff': 0.9}
    agent_high = build_high_level_ghavamzadeh(QLambdaDiscrete, params_high, mdp)

    params_low = {'learning_rate': AdaptiveParameter(value=1e-2)}
    agent_cross = build_low_level_ghavamzadeh(GPOMDP, params_low, mdp)
    agent_plus = build_low_level_ghavamzadeh(GPOMDP, params_low, mdp)

    print('ghavamzadeh')
    res = Parallel(n_jobs=n_jobs)(delayed(ghavamzadeh_experiment)
                                  (mdp, agent_plus, agent_cross, agent_high,
                                   n_epochs, ep_per_epoch, ep_per_eval,
                                   ep_per_run_low_ghavamzadeh)
                                  for _ in range(how_many))
    J, L = parse_joblib(res)
    np.save(subdir + '/J_ghavamzadeh', J)
    np.save(subdir + '/L_ghavamzadeh', L)
