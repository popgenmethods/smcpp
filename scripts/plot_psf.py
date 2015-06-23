from __future__ import division
import numpy as np

def plot_size_functions(pop_size_functions, colors):
    fig, ax = pretty_plot()
    last = []
    tmax = 0.0
    ymax = 0.0
    for (a, b, s), color in zip(pop_size_functions, colors):
        ct = 0.0
        pieces = list(zip(a, b, s))
        for ai, bi, si in pieces[:-1]:
            beta = np.log(bi / ai) / si
            x = np.linspace(ct, ct + si, 20)
            y = ai * np.exp(beta * (x - ct))
            ymax = max(ymax, np.max(y))
            ax.plot(x, y, color=color)
            ct += si
        last.append((ct, pieces[-1]))
        tmax = max(tmax, ct)
    for (ct, (ai, _, _)), color in zip(last, colors):
        ax.plot([ct, tmax * 1.5], [ai, ai], color=color)
    plt.xlim([0, tmax * 1.4])
    plt.ylim([0, ymax * 1.4])
    return plt
