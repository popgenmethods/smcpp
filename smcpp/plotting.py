from __future__ import absolute_import, division, print_function
import json
import matplotlib, matplotlib.cm
matplotlib.use('Agg')
import numpy as np
from numpy import array
import seaborn as sns

import smcpp.defaults
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

def plot_psfs(psfs, xlim, ylim, xlabel, knots=False, logy=False):
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
    for i, (d, off) in enumerate(psfs):
        g = d.get('g') or 1
        if 'b' in d:
            N0 = d['N0']
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
            series.append((fn, x, y, my_axplot, off, N0, g))
        elif 'model' in d:
            cls = getattr(model, d['model']['class'])
            mb = cls.from_dict(d['model'])
            split = False
            if isinstance(mb, model.SMCTwoPopulationModel):
                split = True
                ms = [mb.for_pop(pid) for pid in mb.pids]
                labels = mb.pids
            else:
                ms = [mb]
                labels = [mb.pid]
            for m, l in zip(ms, labels):
                ak = len(smcpp.defaults.additional_knots)
                x = np.cumsum(m.s)
                y = m.stepwise_values().astype('float')
                x = np.insert(x, 0, 0)
                y = np.insert(y, 0, y[0])
                series.append([l, x, y, my_axplot, off, m.N0, g])
                if knots and hasattr(m, '_knots'):
                    knots = m._knots[:-ak]
                    x2, y2 = (knots, np.exp(m[:-ak].astype('float')))
                    # if not logy:
                    #     y *= 1e-3
                    series.append([None, x2, y2, ax.scatter, off, m.N0, g])
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
            series.append((None, x, y, my_axstep, off, N0, g))
    labels = []
    for label, x, y, plotfun, off, N0, g in series:
        xp = 2 * N0 * g * x + off
        yp = N0 * y
        if label is None:
            plotfun(xp, yp, linewidth=2, label=label, color="black")
        else:
            labels += plotfun(xp, yp, label=label, linewidth=2)
        if len(xp) > 2:
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
        xlim = (xmin, xmax)
    if not ylim:
        ylim = (0.0, 1.1 * ymax)
    print("xlim:", xlim)
    ax.set_xlim(*xlim)
    fig.tight_layout()
    return fig, data
