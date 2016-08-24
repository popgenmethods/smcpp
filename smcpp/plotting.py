from __future__ import absolute_import, division, print_function
import json
import matplotlib, matplotlib.cm
matplotlib.use('Agg')
import numpy as np
from numpy import array

from . import model

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

def plot_psfs(psfs, xlim, ylim, xlabel, logy=False):
    fig, ax = pretty_plot()
    xmax = ymax = 0.
    xmin = np.inf
    labels = []
    data = []
    npsf = sum(label != "None" for label, _, _ in psfs)
    colors = list(matplotlib.cm.Dark2(np.linspace(0, 1, npsf)))
    for i, (label, d, off) in enumerate(psfs):
        N0 = d['N0']
        if 'b' in d:
            a = d['a']
            s = d['s']
            b = d['b']
            slope = np.log(a/b) / s
            cum = off
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
            data.append((label, x, y))
            plotfun = ax.plot
        elif 'model' in d:
            m = model.SMCModel.from_dict(d['model'])
            x = np.logspace(np.log10(m.s[0]), np.log10(m.s.sum()), 200)
            y = m(x).astype('float')
            # if not logy:
            #     y *= 1e-3
            data.append((label, x, y))
            plotfun = ax.plot
            x2, y2 = (m._knots, np.exp(m[:].astype('float')))
            x2 *= 2. * d['N0']
            y2 *= d['N0']
            if d['g'] is not None:
                x2 *= d['g']
            ax.scatter(x2,y2)
        else:
            x = np.cumsum(s)
            x = np.insert(x, 0, 0)[:-1]
            y = a
            def f(*args, **kwargs):
                return ax.step(*args, where='post', **kwargs)
            plotfun = f
        x *= 2 * N0
        y *= N0
        # x *= 1. + (i - len(psfs)) / 50.
        if d['g'] is not None:
            x *= d['g']
        x += off
        if label is None:
            plotfun(x, y, linewidth=2, color="black")
        else:
            labels += plotfun(x, y, label=label, color=colors.pop())
        xmin = min(xmin, x[1] * 0.9)
        ymax = max(ymax, np.max(y))
        xmax = max(xmax, np.max(x))
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

def make_psfs(d):
    ret = {'fit': {'a': d['a'], 'b': d['b'], 's': d['s']},
            None: {'a': d['a0'], 'b': d['b0'], 's': d['s0']}}
    if 'coal_times' in d:
        ret['coal_times'] = d['coal_times']
    return ret

def plot_output(psfs, fname, **kwargs):
    save_pdf(plot_psfs(psfs, **kwargs), fname)

def plot_matrices():
    from io import StringIO
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

def imshow(M, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt 
    plt.imshow(M)
    plt.savefig(out_png)

def convert_msmc(msmc_out, fac):
    mu = 1.25e-8
    with open(msmc_out, "rt") as f:
        hdr = next(f)
        a = []
        s = []
        for line in f:
            j, tj, tj1, lam = line.strip().split()
            s.append(float(tj1) - float(tj))
            a.append(float(lam))
    a = np.array(a)
    s = np.array(s)
    s /= mu
    a = 1. / a
    a /= 2. * mu
    return {'a': a / fac, 'b': a / fac, 's': s / fac}

def convert_psmc(psmc_out):
    lines = list(open(psmc_out, "rt"))
    i = -1
    while not lines[i].startswith("RS\t0"):
        i -= 1
    theta0 = float(lines[i - 3].strip().split()[1])
    N0 = theta0 / (4 * 1.25e-8) / 100
    # N0 = 10000.
    t = []
    a = []
    while i < -2:
        fields = lines[i].strip().split()
        t.append(float(fields[2]))
        a.append(float(fields[3]))
        i += 1
    t = np.array(t) 
    a = np.array(a)
    s = t[1:] - t[:-1]
    s = np.append(s, 1.0)
    s *= 3
    a *= 3
    return {'s': s, 'a': a, 'b': a}
