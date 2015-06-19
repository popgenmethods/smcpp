from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

def plot_size_functions(pop_size_functions, colors):
    fig, ax = plt.subplots()
    for (a, b, s) in pop_size_functions:
        ct = 0.0
        for ai, bi, si, col in zip(a, b, s, colors):
            beta = np.log(bi / ai) / si
            x = np.linspace(ct, ct + si, 20)
            y = ai * np.exp(beta * (x - ct))
            ax.plot(x, y, color + "-")
    plt.show()

