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

def plot_psfs(psfs, N0=1e4):
    fig, ax = pretty_plot()
    for a, b, s in psfs:
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
        x = np.concatenate([x, [cum, 1e6]])
        y = np.concatenate([y, [a[-1], a[-1]]])
        ax.plot(x, y)
        # ax.step(cs, a, where='post')
    ax.set_xscale('log')
    ax.set_xlim(100., 1e6)
    ax.set_ylim(0.0, 10.0)
    return fig

def plot_output(out, fname):
    save_pdf(plot_psfs([(out['a0'],out['b0'],out['s0']),(out['a'],out['b'],out['s'])]), fname)
    
