from __future__ import absolute_import, division, print_function
import json
import matplotlib, matplotlib.style, matplotlib.cm

matplotlib.use("Agg")
matplotlib.style.use("seaborn-ticks")
import numpy as np
from numpy import array
from collections import defaultdict

import smcpp.defaults
from . import model


def pretty_plot():
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure

    fig = Figure()
    FigureCanvas(fig)
    ax = fig.add_subplot(111)
    return fig, ax


def plot_psfs(psfs, xlim, ylim, xlabel, knots=False, logy=False, stats={}):
    fig, ax = pretty_plot()
    xmax = ymax = 0.
    xmin = ymin = np.inf
    labels = []
    series = []
    data = [["label", "x", "y", "plot_type", "plot_num"]]

    def saver(f, ty):

        def g(x, y, label, data=data, **kwargs):
            data += [(label, xx, yy, ty, saver.plot_num) for xx, yy in zip(x, y)]
            saver.plot_num += 1
            if label not in g.seen:
                g.seen.append(label)
                kwargs["label"] = label
            return f(x, y, **kwargs)

        g.i = 0
        g.seen = []
        return g

    saver.plot_num = 0
    my_axplot = saver(ax.plot, "path")
    my_axstep = saver(ax.step, "step")
    vlines = []
    models = []
    for i, (d, off) in enumerate(psfs):
        g = d.get("g") or 1
        if "b" in d:
            N0 = d["N0"]
            a = d["a"]
            s = d["s"]
            b = d["b"]
            slope = np.log(a / b) / s
            cum = 0.
            x = []
            y = []
            for aa, bb, ss in zip(b[:-1], slope[:-1], s[:-1]):
                tt = np.linspace(cum, cum + ss, 200)
                yy = aa * np.exp(bb * (cum + ss - tt))
                x = np.concatenate([x, tt])
                y = np.concatenate([y, yy])
                cum += ss
            x = np.concatenate([x, [cum, 2 * cum]])
            y = np.concatenate([y, [a[-1], a[-1]]])
            # if not logy:
            #     y *= 1e-3
            series.append((fn, x, y, my_axplot, off, N0, g))
        elif "model" in d:
            cls = getattr(model, d["model"]["class"])
            mb = cls.from_dict(d["model"])
            models.append(mb)
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
                y = m.stepwise_values().astype("float")
                x = np.insert(x, 0, 0)
                y = np.insert(y, 0, y[0])
                if split and l == mb.pids[-1]:
                    vlines.append(mb.split * 2 * m.N0 * g)
                    xf = x < mb.split
                    x = x[xf]
                    x = np.r_[x, mb.split]
                    y = y[xf]
                    y = np.r_[y, y[-1]]
                    split = False
                series.append([l, x, y, my_axplot, off, m.N0, g])
                if knots and hasattr(m, "_knots"):
                    knots = m._knots[:-ak]
                    x2, y2 = (knots, np.exp(m[:-ak].astype("float")))
                    # if not logy:
                    #     y *= 1e-3
                    series.append([None, x2, y2, ax.scatter, off, m.N0, g])
        else:
            x = np.cumsum(d["s"])
            x = np.insert(x, 0, 0)[:-1]
            y = d["a"]
            series.append((None, x, y, my_axstep, off, N0, g))
    for statname in stats:
        magg = model.aggregate(*models, stat=stats[statname])
        series.append(
            [
                statname,
                np.cumsum(magg.s),
                magg.stepwise_values().astype("float"),
                my_axplot,
                0.,
                magg.N0,
                g,
            ]
        )
    labels = []
    NUM_COLORS = len({label for label, *_ in series})
    cm = matplotlib.cm.get_cmap("gist_rainbow")
    COLORS = [cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)]
    label_colors = defaultdict(lambda: COLORS[len(label_colors)])
    for label, x, y, plotfun, off, N0, g in series:
        xp = 2 * N0 * g * x + off
        yp = N0 * y
        if label is None:
            plotfun(xp, yp, linewidth=2, label=label, color="black")
        else:
            labels += plotfun(
                xp, yp, label=label, linewidth=2, color=label_colors[label]
            )
        if len(xp) > 2:
            xmin = min(xmin, xp[1] * 0.9)
        ymin = min(ymin, np.min(yp))
        ymax = max(ymax, np.max(yp))
        xmax = max(xmax, np.max(xp))
    if labels:
        first_legend = ax.legend(handles=labels, loc=9, ncol=4, prop={"size": 8})
    for x in vlines:
        ax.axvline(x)
    ax.set_xscale("log")
    ax.set_ylabel(r"$N_e$")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    if not xlim:
        xlim = (xmin, xmax)
    if not ylim:
        ylim = (.9 * ymin, 1.1 * ymax)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    fig.tight_layout()
    return fig, data
