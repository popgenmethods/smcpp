from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


themes = """Palette	Palette Column	Palette Row	PC (c,r)	RGB	One
Tableau 10	1	2	T 10: 1,2	255.127.14	1
Tableau 10	1	4	T 10: 1,4	214.39.40	1
Tableau 10	1	5	T 10: 1,5	148.103.189	1
Tableau 10	1	3	T 10: 1,3	44.160.44	1
Tableau 10	1	1	T 10: 1,1	31.119.180	1
Tableau 10	2	2	T 10: 2,2	227.119.194	1
Tableau 10	2	4	T 10: 2,4	188.189.34	1
Tableau 10	2	1	T 10: 2,1	140.86.75	1
Tableau 10	2	3	T 10: 2,3	127.127.127	1
Tableau 10	2	5	T 10: 2,5	23.190.207	1"""

palette = [(.2, .2, .2)]
for row in themes.split("\n")[1:]:
    fields = row.strip().split("\t")
    rgb = tuple([int(x) / 255. for x in fields[-2].split(".")])
    palette.append(rgb)

def pretty_plot():
    fig, ax = plt.subplots()
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)  
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left()  
    return fig, ax

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
