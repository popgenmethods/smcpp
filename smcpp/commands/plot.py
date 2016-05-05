import os.path
import sys
import argparse
import numpy as np
import itertools as it
from .. import util, plotting
from ..estimation_result import EstimationResult

def init_parser(parser):
    parser.add_argument("-g", type=float, help="Plot x-axis in years assuming a generation time of g")
    parser.add_argument("--logy", action="store_true", help="ploy y on log axis")
    parser.add_argument("-t", "--offsets", type=float, nargs="+", 
            help="list of offsets, one for each <model>, to shift x axes. Useful for plotting aDNA")
    parser.add_argument("-x", "--xlim", type=float, nargs=2, default=None, help="x-axis limits")
    parser.add_argument("-y", "--ylim", type=float, nargs=2, default=None, help="y-axis limits")
    parser.add_argument("-l", "--labels", type=str, help="label for each plotted function", nargs="+")
    parser.add_argument("-c", "--csv", help="also write <plot.pdf.csv> containing data used to make plot", action="store_true", default=False)
    parser.add_argument("pdf", type=str, help="output PDF", metavar="plot.pdf", widget="FileChooser")
    parser.add_argument("model", type=str, help="SMC++ models to plot", nargs="+", widget="MultiFileChooser")

def main(args):
    psfs = []
    if args.labels is None:
        args.labels = []
    if args.offsets is None:
        args.offsets = []
    for fn, label, off in it.izip_longest(args.model, args.labels, args.offsets, fillvalue=None):
        if not os.path.exists(fn):
            sys.exit("File not found: %s" % fn)
        if label == "None": 
            label = None
        if fn in ["human", "sawtooth"]:
            p = getattr(util, fn)
            d = {k: p[k] for k in "abs"}
            d['N0'] = p['N0']
        else:
            er = EstimationResult.load(fn)
            d = dict(zip("abs", np.array(er.model)))
            d['N0'] = er.N0
        if args.g is not None:
            d['s'] *= args.g
        psfs.append((label, d, off or 0))
    fig, data = plotting.plot_psfs(psfs, xlim=args.xlim, ylim=args.ylim,
                                   xlabel="Generations" if args.g is None else "Years", 
                                   logy=args.logy) 
    plotting.save_pdf(fig, args.pdf)
    if args.csv:
        with open(args.pdf + ".csv", "wt") as f:
            f.write("label,x,y\n")
            f.write("\n".join(",".join((str(label), str(xx), str(yy))) for label, x, y in data for xx, yy in zip(x, y)))
