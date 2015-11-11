import matplotlib
matplotlib.use('Agg')

import numpy as np
from numpy import array

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
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure
    fig = Figure()
    FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)  
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left()  
    return fig, ax

def save_pdf(plt, filename):
    from matplotlib.backends.backend_pdf import PdfPages
    pp = PdfPages(filename)
    plt.savefig(pp, format='pdf')
    pp.close()

def plot_psfs(psfs, N0=1e4, xlim=None, ylim=None, order=None):
    fig, ax = pretty_plot()
    xmax = ymax = 0.
    labels = []
    if order is None:
        order = sorted(psfs)
    for label in order:
        a = psfs[label]['a']
        b = psfs[label]['b']
        s = psfs[label]['s']
        sp = s * 25.0 * 2 * N0
        # cs = np.concatenate(([100.], np.cumsum(s) * 25.0 * 2 * N0))
        # cs[-1] = 1e7
        # a = np.concatenate((a, [a[-1]]))
        # b = np.concatenate((b, [a[-1]]))
        slope = np.log(b/a) / sp
        cum = 0.
        x = []
        y = []
        for aa, bb, ss in zip(a[:-1], slope[:-1], sp[:-1]):
            tt = np.linspace(cum, cum + ss, 100)
            yy = aa * np.exp(bb * (tt - cum))
            x = np.concatenate([x, tt])
            y = np.concatenate([y, yy])
            cum += ss
        x = np.concatenate([x, [cum, 2 * cum]])
        y = np.concatenate([y, [a[-1], a[-1]]])
        if label is None:
            ax.plot(x, y, linewidth=2, color="black")
        else:
            labels += ax.plot(x, y, label=label)
        # ax.step(cs, a, where='post')
        ymax = max(ymax, np.max(y))
        xmax = max(xmax, np.max(x))
    first_legend = ax.legend(handles=labels, loc=1)
    ax.set_xscale('log')
    if not xlim:
        xlim = (3000., 1.1 * xmax)
    if not ylim:
        ylim=(0.0, 1.1 * ymax)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    return fig

def make_psfs(d):
    ret = {'fit': {'a': d['a'], 'b': d['b'], 's': d['s']},
            None: {'a': d['a0'], 'b': d['b0'], 's': d['s0']}}
    return ret

def plot_output(psfs, fname, **kwargs):
    save_pdf(plot_psfs(psfs, **kwargs), fname)

def plot_matrices():
    from cStringIO import StringIO
    mats = {}
    with open("matrices.txt", "rt") as mfile:
        try:
            while True:
                name = next(mfile).strip()
                lines = []
                while True:
                    l = next(mfile).strip()
                    if l:
                        lines.append(l)
                    else:
                        break
                mats[name] = np.loadtxt(StringIO("\n".join(lines)))
        except StopIteration:
            mats[name] = np.loadtxt(StringIO("\n".join(lines)))
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure
    for name in mats:
        fig = Figure()
        FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.pcolor(np.log(1 + np.abs(mats[name])))
        save_pdf(fig, "%s.pdf" % name)
    return mats
