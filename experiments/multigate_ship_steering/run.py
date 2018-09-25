import datetime
from joblib import Parallel, delayed

from mushroom.algorithms.policy_search import *
from mushroom.algorithms.value.td import *
from mushroom.utils.parameters import AdaptiveParameter, Parameter, \
    ExponentialDecayParameter
from mushroom.utils.folder import *

from mushroom_hierarchical.environments.ship_steering_multiple_gates import *

from hierarchical import *


if __name__ == '__main__':

    n_jobs = -1

    how_many = 1#00
    n_epochs = 50
    ep_per_epoch_train = 20
    ep_per_epoch_eval = 50
    n_iterations = 10

    ep_per_fit_low = 10
    ep_per_fit_mid = 10

    mask_high_epochs = 10

    # MDP
    mdp = ShipSteeringMultiGate(n_steps_action=3, small=True)

    # directory
    name = 'multigate_ship_steering'
    subdir = name + '_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')\
             + '/'

    mk_dir_recursive('./' + subdir)
    force_symlink('./' + subdir, name + '_latest')

    # Hierarchical
    algs_and_params_hier = [
        (QLearning, {'learning_rate': Parameter(value=0.5)},
         GPOMDP, {'learning_rate': AdaptiveParameter(value=20) if mdp.small else AdaptiveParameter(value=50)},
         PGPE, {'learning_rate': AdaptiveParameter(value=5e-4)})
         ]

    for alg_h, params_h, alg_m, params_m, alg_l, params_l \
            in algs_and_params_hier:
        epsilon = ExponentialDecayParameter(value=1.0, decay_exp=0.5)
        agent_h = build_high_level_agent(alg_h, params_h, mdp, epsilon)

        mu = 250 if mdp.small else 500
        sigma = 125 if mdp.small else 250
        agent_m0 = build_mid_level_agent(alg_m, params_m, mdp, mu, sigma)
        agent_m1 = build_mid_level_agent(alg_m, params_m, mdp, mu, sigma)
        agent_m3 = build_mid_level_agent(alg_m, params_m, mdp, mu, sigma)
        agent_m4 = build_mid_level_agent(alg_m, params_m, mdp, mu, sigma)

        agent_l = build_low_level_agent(alg_l, params_l, mdp)

        ep_per_run_hier = ep_per_epoch_train // n_iterations

        print('High: ', alg_h.__name__, ' Medium: ', alg_m.__name__ ,
              ' Low: ', alg_l.__name__)
        J = Parallel(n_jobs=n_jobs)(delayed(hierarchical_experiment)
                                    (mdp, agent_l,
                                     agent_m0, agent_m1,
                                     agent_m3, agent_m4,
                                     agent_h, n_epochs,
                                     n_iterations, ep_per_epoch_train,
                                     ep_per_epoch_eval, ep_per_fit_low,
                                     ep_per_fit_mid, mask_high_epochs)
                                    for _ in range(how_many))
        np.save(subdir + '/H_' + alg_h.__name__ + '_' + alg_l.__name__, J)