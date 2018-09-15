import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from matplotlib2tikz import save as tikz_save


def get_mean_and_confidence(data):
    mean = np.mean(data, axis=0)
    se = st.sem(data, axis=0)
    n = len(data)

    _, interval = st.t.interval(0.95, n-1, scale=se)

    return mean, interval


def create_plot(algs, colors, dictionary, y_label, legend=False,
                x_label='epoch', logarithmic=False, output_dir='',
                plot_name='figure'):
    plt.figure()

    if logarithmic:
        plt.yscale('symlog')

    for alg, c in zip(algs, colors):
        (mean, err) = dictionary[alg]
        plt.errorbar(x=np.arange(len(mean)), y=mean, yerr=err, color=c)

    if legend:
        plt.legend(algs)

    plt.xlabel(x_label)
    plt.ylabel(y_label)


    if output_dir:
        tikz_save(output_dir + '/' + plot_name + '.tex',
                  figureheight='\\figureheight',
                  figurewidth='\\figurewidth')
