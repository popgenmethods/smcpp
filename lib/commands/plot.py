import argparse
import numpy as np
from .. import util, plotting

def init_parser(parser):
    parser.add_argument("-g", type=float, default=25.0, help="generation time")
    parser.add_argument("--xlim", type=str, default="100,1000000", help="xmin,xmax pair")
    parser.add_argument("--ylim", type=str, default=None, help="ymin,ymax pair")
    parser.add_argument("pdf", type=str, help="output PDF")
    parser.add_argument("out", type=str, help="<label:filename> pairs", nargs="+")

def main(args):
    psfs = {}
    if args.xlim is not None:
        args.xlim = tuple(map(float, args.xlim.split(",")))
    if args.ylim is not None:
        args.ylim = tuple(map(float, args.ylim.split(",")))
    for pair in args.out:
        label, fn = pair.split(":")
        if label == "None": 
            label = None
        if fn in ["human", "sawtooth"]:
            psfs[label] = getattr(util, fn)
            psfs[label]['a'] *= 20000.
            psfs[label]['b'] *= 20000.
            psfs[label]['s'] = np.cumsum(psfs[label]['s_gen'])
        else:
            psfs[label] = dict(zip("abs", np.loadtxt(fn).T))
        psfs[label]['s'] *= args.g
    plotting.save_pdf(plotting.plot_psfs(psfs, args.xlim, args.ylim), args.pdf)

class PlotCommand:
    def self.init_parser(klass, parser):
        init_parser(parser)
        parser.set_defaults(func=klass())
    def __call__(self, args):
        main(args)
