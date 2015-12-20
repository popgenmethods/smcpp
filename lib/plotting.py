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

def plot_psfs(psfs, N0=1e4, xlim=None, ylim=None, order=None):
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
    
if __name__ == "1__main__":
    N0 = 10000.
    sawtooth = {
            'a': np.array([7.1, 7.1, 0.9, 7.1, 0.9, 7.1, 0.9]),
            'b': np.array([7.1, 0.9, 7.1, 0.9, 7.1, 0.9, 0.9]),
            's': np.array([1000.0, 4000.0 - 1000., 10500. - 4000., 65000. - 10500., 115000. - 65000., 1e6 - 115000, 1.0]) / 25.0 / (2 * N0)
            }
    smc10 = {'a': array([ 11.385128,   6.411377,   5.521558,   5.612072,   6.081421,   6.43935 ,   6.268385,   5.581489,   4.862248,
        4.230269,   3.84875 ,   3.409132,   2.753648,   1.667301,   1.298603,   1.342306,   2.044066,   2.869721,
        3.967727,   5.257937,   6.368057,   6.584663,   6.31398 ,   5.400804,   4.257609,   3.421085,   2.45475 ,
        1.576378,   1.171175,   0.902922]), 'a0': array([ 7.1,  7.1,  0.9,  7.1,  0.9,  7.1,  0.9]), 's': array([ 0.01    ,  0.002044,  0.002462,  0.002965,  0.003571,  0.004301,  0.00518 ,  0.006239,  0.007515,  0.009051,
            0.010901,  0.013129,  0.015813,  0.019045,  0.022938,  0.027626,  0.033273,  0.040074,  0.048266,  0.058132,
            0.070014,  0.084326,  0.101562,  0.122322,  0.147326,  0.17744 ,  0.21371 ,  0.257394,  0.310007,  0.373375]), 'b': array([ 11.385128,   6.411377,   5.521558,   5.612072,   6.081421,   6.43935 ,   6.268385,   5.581489,   4.862248,
                4.230269,   3.84875 ,   3.409132,   2.753648,   1.667301,   1.298603,   1.342306,   2.044066,   2.869721,
                3.967727,   5.257937,   6.368057,   6.584663,   6.31398 ,   5.400804,   4.257609,   3.421085,   2.45475 ,
                1.576378,   1.171175,   0.902922]), 'b0': array([ 7.1,  0.9,  7.1,  0.9,  7.1,  0.9,  0.9]), 't_start': 1447232940.921908, 's0': array([ 0.002   ,  0.006   ,  0.013   ,  0.109   ,  0.1     ,  1.77    ,  0.000002]), 't_now': 1447239193.91054, 'argv': ['scripts/em.py', '10', '10', '5e7']}
    smc25 = {'a': array([ 5.081572,  0.437756,  2.543732,  4.533135,  6.282798,  6.830099,  7.028583,  6.727999,  5.718671,  4.426326,
        3.908583,  3.378027,  2.876592,  1.77071 ,  1.207451,  1.347225,  2.040817,  2.831229,  3.854859,  5.330478,
        6.351079,  6.546177,  6.162707,  5.338964,  4.224983,  3.372615,  2.4785  ,  1.548987,  1.152883,  0.894789]), 'a0': array([ 7.1,  7.1,  0.9,  7.1,  0.9,  7.1,  0.9]), 's': array([ 0.01    ,  0.002044,  0.002462,  0.002965,  0.003571,  0.004301,  0.00518 ,  0.006239,  0.007515,  0.009051,
        0.010901,  0.013129,  0.015813,  0.019045,  0.022938,  0.027626,  0.033273,  0.040074,  0.048266,  0.058132,
        0.070014,  0.084326,  0.101562,  0.122322,  0.147326,  0.17744 ,  0.21371 ,  0.257394,  0.310007,  0.373375]), 'b': array([ 5.081572,  0.437756,  2.543732,  4.533135,  6.282798,  6.830099,  7.028583,  6.727999,  5.718671,  4.426326,
        3.908583,  3.378027,  2.876592,  1.77071 ,  1.207451,  1.347225,  2.040817,  2.831229,  3.854859,  5.330478,
        6.351079,  6.546177,  6.162707,  5.338964,  4.224983,  3.372615,  2.4785  ,  1.548987,  1.152883,  0.894789]), 'b0': array([ 7.1,  0.9,  7.1,  0.9,  7.1,  0.9,  0.9]), 't_start': 1447232970.385792, 's0': array([ 0.002   ,  0.006   ,  0.013   ,  0.109   ,  0.1     ,  1.77    ,  0.000002]), 't_now': 1447239288.338471, 'argv': ['scripts/em.py', '25', '10', '5e7']}
    smc50 = {'s':np.array([0.01,0.002044,0.002462,0.002965,0.003571,0.004301,0.00518,0.006239,0.007515,0.009051,0.010901,0.013129
    ,0.015813,0.019045,0.022938,0.027626,0.033273,0.040074,0.048266,0.058132,0.070014,0.084326,0.101562,0.122322
    ,0.147326,0.17744,0.21371,0.257394,0.310007,0.373375]),
    'a':np.array([5.01, .9, 4.0, 5.05, 5.8, 5.9, 5.7, 5.3, 5.1, 4.8, 4.2, 3.5, 3.3, 3.1,
    0.022938,0.027626,0.033273,0.040074,0.048266,0.058132,0.070014,0.084326,0.101562,0.122322
    ,0.147326,0.17744,0.21371,0.257394,0.310007,0.373375])}
    smc50['b'] = smc50['a']
    msmc = convert_msmc("/scratch/terhorst/datasets/2/msmc/msmc_out.final.txt", 2 * N0)
    msmc['s'] /= 2.0
    psmc = convert_psmc("/scratch/terhorst/datasets/2/psmc/psmc.out")
    psfs = {
            None: sawtooth,
            'PSMC': psmc,
            'MSMC': msmc,
            'SMC++ (n=10)': smc10,
            'SMC++ (n=25)': smc25,
            }
    save_pdf(plot_psfs(psfs, xlim=(1000, 1e6), ylim=(0, 15)), "/export/home/terhorst/Dropbox.new/Dropbox/di.pdf")
