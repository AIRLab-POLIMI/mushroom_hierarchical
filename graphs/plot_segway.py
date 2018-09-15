from mushroom.utils.folder import mk_dir_recursive
from common import *


base_dir = '../results/segway'
output_dir = '../out/segway'
mk_dir_recursive(output_dir)

algs = [
        'REPS',
        'RWR',
        'H_RWR_RWR',
        'H_REPS_RWR']

colors = ['b', 'r', 'g', 'c', 'm']

J_results = dict()
L_results = dict()

for alg in algs:
    J = np.load(base_dir + '/J_' + alg + '.npy')
    J_results[alg] = get_mean_and_confidence(J)
    print(alg, ': ', J.shape)

    L = np.load(base_dir + '/L_' + alg + '.npy')
    L_results[alg] = get_mean_and_confidence(L)

create_plot(algs, colors, J_results, 'J', legend=True, logarithmic=True,
            output_dir=output_dir, plot_name='J')
create_plot(algs, colors, L_results, 'L', legend=False,
            output_dir=output_dir, plot_name='L')

plt.show()




