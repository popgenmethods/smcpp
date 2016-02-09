from __future__ import division
import matplotlib
matplotlib.use('Agg')

import numpy as np
from numpy import array

sawtooth = {
        'a': np.array([7.1, 7.1, 0.9, 7.1, 0.9, 7.1, 0.9]),
        'b': np.array([7.1, 0.9, 7.1, 0.9, 7.1, 0.9, 0.9]),
        's': np.array([1000.0, 4000.0 - 1000., 10500. - 4000., 65000. - 10500., 115000. - 65000., 1e6 - 115000, 1.0]) / 25.0 /
        (2 * 10000.)
        }

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

def plot_psfs(psfs, xlim=None, ylim=None, order=None):
    fig, ax = pretty_plot()
    xmax = ymax = 0.
    labels = []
    if order is None:
        order = sorted(psfs)
    for label in order:
        if label == "coal_times": continue
        a = psfs[label]['a']
        b = psfs[label]['b']
        s = psfs[label]['s']
        s = np.concatenate([[0.], s])
        s = s[1:] - s[:-1]
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
        if label is None:
            ax.plot(x, y, linewidth=2, color="black")
        else:
            labels += ax.plot(x, y, label=label)
        # ax.step(cs, a, where='post')
        ymax = max(ymax, np.max(y))
        xmax = max(xmax, np.max(x))
    if 'coal_times' in psfs:
        from scipy.stats import gaussian_kde
        ct = psfs['coal_times']
        del psfs['coal_times']
        kde = gaussian_kde(ct)
        x = np.arange(xlim[0], xlim[1], 5000)
        y = kde.evaluate(x)
        y *= 10. / max(y)
        ax.plot(x, y, color="grey", linestyle="--")
    first_legend = ax.legend(handles=labels, loc=1)
    ax.set_xscale('log')
    ax.set_xlabel('Years')
    ax.set_ylabel(r'$N_e \times 10^3$')
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
    if 'coal_times' in d:
        ret['coal_times'] = d['coal_times']
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
