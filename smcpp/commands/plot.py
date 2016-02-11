import argparse
import numpy as np
import itertools as it
from .. import util, plotting
from ..model import SMCModel

def init_parser(parser):
    parser.add_argument("--g", type=float, help="Plot x-axis in years assuming a generation time of g")
    parser.add_argument("--xlim", type=str, default="100,1000000", help="xmin,xmax pair")
    parser.add_argument("--ylim", type=str, default=None, help="ymin,ymax pair")
    parser.add_argument("--labels", type=str, help="labels for each plotted function", nargs="+")
    parser.add_argument("pdf", type=str, help="output PDF", widget="FileChooser")
    parser.add_argument("model", type=str, help="SMC++ models to plot", nargs="+", widget="MultiFileChooser")

def main(args):
    psfs = {}
    if args.xlim is not None:
        args.xlim = tuple(map(float, args.xlim.split(",")))
    if args.ylim is not None:
        args.ylim = tuple(map(float, args.ylim.split(",")))
    for fn, label in it.izip_longest(args.model, args.labels, fillvalue=None):
        if label == "None": 
            label = None
        if fn in ["human", "sawtooth"]:
            p = getattr(util, fn)
            psfs[label] = {k: p[k] * p['N0'] for k in "abs"}
        else:
            try:
                m = SMCModel.from_file(fn)
                # Plot size history in diploids => do not scale a,b by 2
                psfs[label] = {'a': m.a * m.N0, 'b': m.b * m.N0, 's': 2. * np.cumsum(m.s) * m.N0}
            except ValueError:
                # Try old csv format. Here everything should have correct scaling
                m = np.array([[float(x) for x in line.strip().split(",")] for line in open(fn, "rt")]).T
                psfs[label] = dict(zip('abs', m))
        if args.g is not None:
            psfs[label]['s'] *= args.g
    plotting.save_pdf(plotting.plot_psfs(psfs, xlim=args.xlim, ylim=args.ylim, 
        xlabel="Generations" if args.g is None else "Years"), args.pdf)
