from __future__ import absolute_import, division, print_function
import json
import matplotlib, matplotlib.cm
matplotlib.use('Agg')
import numpy as np
from numpy import array
import seaborn as sns

from . import model

def pretty_plot():
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure
    sns.set(style="ticks")
    fig = Figure()
    FigureCanvas(fig)
    ax = fig.add_subplot(111)
    sns.despine(fig)
    return fig, ax

def save_pdf(plt, filename):
    from matplotlib.backends.backend_pdf import PdfPages
    pp = PdfPages(filename)
    plt.savefig(pp, format='pdf')
    pp.close()

def plot_psfs(psfs, xlim, ylim, xlabel, logy=False):
    fig, ax = pretty_plot()
    xmax = ymax = 0.
    xmin = np.inf
    labels = []
    series = []
    data = [["label", "x", "y", "plot_type", "plot_num"]]
    def saver(f, ty):
        def g(x, y, label, data=data, **kwargs):
            data += [(label, xx, yy, ty, saver.plot_num) for xx, yy in zip(x, y)]
            saver.plot_num += 1
            return f(x, y, label=label, **kwargs)
        g.i = 0
        return g
    saver.plot_num = 0
    my_axplot = saver(ax.plot, "path")
    my_axstep = saver(ax.step, "step")
    npsf = sum(label != "None" for label, _, _ in psfs)
    for i, (label, d, off) in enumerate(psfs):
        N0 = d['N0']
        g = d.get('g', None) or 1
        if 'b' in d:
            a = d['a']
            s = d['s']
            b = d['b']
            slope = np.log(a/b) / s
            cum = 0.
            x = []
            y = []
            for aa, bb, ss in zip(b[:-1], slope[:-1], s[:-1]):
                tt = np.linspace(cum, cum + ss, 100)
                yy = aa * np.exp(bb * (cum + ss - tt))
                x = np.concatenate([x, tt])
                y = np.concatenate([y, yy])
                cum += ss
            x = np.concatenate([x, [cum, 2 * cum]])
            y = np.concatenate([y, [a[-1], a[-1]]])
            # if not logy:
            #     y *= 1e-3
            series.append((label, x, y, my_axplot, off, N0, g))
        elif 'model' in d:
            cls = getattr(model, d['model']['class'])
            mb = cls.from_dict(d['model'])
            split = False
            if isinstance(mb, model.SMCTwoPopulationModel):
                split = True
                ms = mb.splitted_models()
                if label:
                    labels = label.split(",")
                else:
                    labels = (None, None)
            else:
                ms = [mb]
                labels = [label]
            for m, l in zip(ms, labels):
                x = np.logspace(np.log10(m.s[0]), np.log10(m.s.sum()), 200)
                y = m(x).astype('float')
                x2, y2 = (m._knots, np.exp(m[:].astype('float')))
                # if not logy:
                #     y *= 1e-3
                series.append([l, x, y, my_axplot, off, N0, g])
                series.append([None, x2, y2, ax.scatter, off, N0, g])
            if split:
                for i in 1, 2:
                    x = series[-i][1]
                    coords = x <= mb.split
                    for j in 1, 2:
                        series[-i][j] = series[-i][j][coords]
        else:
            x = np.cumsum(d['s'])
            x = np.insert(x, 0, 0)[:-1]
            y = d['a']
            series.append((label, x, y, my_axstep, off, N0, g))
    labels = []
    for label, x, y, plotfun, off, N0, g in series:
        xp = 2 * N0 * g * x + off
        yp = N0 * y
        if label is None:
            plotfun(xp, yp, linewidth=2, label=label, color="black")
        else:
            labels += plotfun(xp, yp, label=label, linewidth=2)
        xmin = min(xmin, xp[1] * 0.9)
        ymax = max(ymax, np.max(yp))
        xmax = max(xmax, np.max(xp))
    if labels:
        first_legend = ax.legend(handles=labels, loc=9, ncol=4, prop={'size':8})
    ax.set_xscale('log')
    ax.set_ylabel(r'$N_e$')
    if logy:
        ax.set_yscale('log')
    ax.set_xlabel(xlabel)
    if not xlim:
        xlim = (0.9 * xmin, 1.1 * xmax)
    if not ylim:
        ylim=(0.0, 1.1 * ymax)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    fig.tight_layout()
    return fig, data
