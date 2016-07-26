import os.path
import sys
import argparse
import numpy as np
import itertools as it
import json
from .. import util, plotting, model

def init_parser(parser):
    parser.add_argument("-g", nargs="+", type=float, help="Plot x-axis in years assuming generation time(s) of g")
    parser.add_argument("--logy", action="store_true", help="ploy y on log axis")
    parser.add_argument("-t", "--offsets", type=float, nargs="+", 
            help="list of offsets, one for each <model>, to shift x axes. Useful for plotting aDNA")
    parser.add_argument("--median", action="store_true", help="plot median and iqr")
    parser.add_argument("-x", "--xlim", type=float, nargs=2, default=None, help="x-axis limits")
    parser.add_argument("-y", "--ylim", type=float, nargs=2, default=None, help="y-axis limits")
    parser.add_argument("-l", "--labels", type=str, help="label for each plotted function", nargs="+")
    parser.add_argument("-c", "--csv", help="also write <plot.pdf.csv> containing data used to make plot", action="store_true", default=False)
    parser.add_argument("pdf", type=str, help="output PDF", metavar="plot.pdf", widget="FileChooser")
    parser.add_argument("model", type=str, help="SMC++ models to plot", nargs="+", widget="MultiFileChooser")

def main(args):
    psfs = []
    if args.g is not None and len(args.g) == 1:
        args.g *= len(args.model)
    if args.offsets is not None and len(args.offsets) == 1:
        args.offsets *= len(args.model)
    if args.labels is None:
        args.labels = []
    if args.offsets is None:
        args.offsets = []
    for fn, label, off, g in it.izip_longest(args.model, args.labels, args.offsets, args.g, fillvalue=None):
        if label == "None": 
            label = None
        if fn in ["human", "sawtooth"]:
            p = getattr(util, fn)
            d = {k: p[k] for k in "abs"}
            d['N0'] = p['N0']
        elif not os.path.exists(fn):
            sys.exit("File not found: %s" % fn)
        else:
            res = json.load(open(fn, "rt"))
            mod = res['model']
            klass = getattr(model, mod['class'])
            m = klass.from_dict(mod)
            a = m.stepwise_values().astype('float')
            s = m.s
            d = {'a': m.stepwise_values(), 's': m.s, 'N0': res['N0']}
        if g is not None:
            d['s'] *= g
        psfs.append((label, d, off or 0))
    if args.median:
        dmed = {'s': psfs[-1][1]['s'], 'N0': psfs[-1][1]['N0']}
        for x in "ab":
            dmed[x] = np.median([p[1][x] for p in psfs], axis=0)
        psfs.append((None, dmed, 0))
    fig, data = plotting.plot_psfs(psfs, xlim=args.xlim, ylim=args.ylim,
                                   xlabel="Generations" if args.g is None else "Years", 
                                   logy=args.logy) 
    plotting.save_pdf(fig, args.pdf)
    if args.csv:
        with open(args.pdf + ".csv", "wt") as f:
            f.write("label,x,y\n")
            f.write("\n".join(",".join((str(label), str(xx), str(yy))) for label, x, y in data for xx, yy in zip(x, y)))
