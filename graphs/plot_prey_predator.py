from mushroom.utils.folder import mk_dir_recursive
from common import *


base_dir = '../results/prey_predator'
output_dir = '../out/prey_predator'
mk_dir_recursive(output_dir)

algs = ['H_DQN_GPOMDP']

colors = ['b', 'r', 'g', 'c', 'm']

J_results = dict()
L_results = dict()

for alg in algs:
    # J
    J = np.load(base_dir + '/J_' + alg + '.npy')
    J_results[alg] = get_mean_and_confidence(J)
    print(alg, ' J: ', J.shape)

    # L
    L = np.load(base_dir + '/L_' + alg + '.npy')
    print(alg, ' L: ', L.shape)
    L_results[alg] = get_mean_and_confidence(L)

create_plot(algs, colors, J_results, 'J', legend=True,
            output_dir=output_dir, plot_name='J')
create_plot(algs, colors, L_results, 'L', legend=False,
            output_dir=output_dir, plot_name='L')

plt.show()




