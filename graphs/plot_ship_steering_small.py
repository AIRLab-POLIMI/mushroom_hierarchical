from common import *
from mushroom.utils.folder import mk_dir_recursive


base_dir = '../results/ship_steering_small'
output_dir = '../out/ship_steering_small'
mk_dir_recursive(output_dir)

algs = ['GPOMDP',
        'PGPE',
        'REPS',
        'RWR',
        'H_GPOMDP_PGPE']

colors = ['b', 'r', 'g', 'c', 'm', 'k']

J_results = dict()
L_results = dict()

for alg in algs:
    J = np.load(base_dir + '/J_' + alg + '.npy')
    J_results[alg] = get_mean_and_confidence(J)

    L = np.load(base_dir + '/L_' + alg + '.npy')
    L_results[alg] = get_mean_and_confidence(L)

create_plot(algs, colors, J_results, 'J', legend=True,
            output_dir=output_dir, plot_name='J')
create_plot(algs, colors, L_results, 'L', legend=False,
            output_dir=output_dir, plot_name='L')

plt.show()




