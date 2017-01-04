import os.path
import sys
import argparse
import numpy as np
import itertools as it
import json
import csv
from six.moves import zip_longest
from .. import util, plotting, model

from . import command


class Plot(command.Command):
    "Plot size history from fitted model"

    def __init__(self, parser):
        parser.add_argument("-g", type=float,
                            help="Plot x-axis in years assuming generation time(s) of g")
        parser.add_argument("-s", "--step-function", action="store_true",
                            help="Plot the step function used to represent the spline "
                                 "instead of spline itself.")
        parser.add_argument("--csv", "-c", action="store_true",
                            help="Also output <plot.csv> containing the plotted points")
        parser.add_argument("--logy", action="store_true",
                            help="ploy y on log axis")
        parser.add_argument("-t", "--offsets", type=float, nargs="+",
                            help="list of offsets, one for each <model>, to shift x axes. "
                                 "Mainly useful for plotting aDNA")
        # parser.add_argument("--median", action="store_true",
        #                     help="plot median and iqr")
        parser.add_argument("-x", "--xlim", type=float, nargs=2,
                            default=None, help="x-axis limits")
        parser.add_argument("-y", "--ylim", type=float, nargs=2,
                            default=None, help="y-axis limits")
        # parser.add_argument("-l", "--labels", type=str,
        # help="label for each plotted function", nargs="+")
        parser.add_argument("out", type=str, help="output image",
                            metavar="plot.[pdf|png|gif|jpeg]")
        parser.add_argument("model", type=str,
                            help="SMC++ models to plot", nargs="+")

    def main(self, args):
        psfs = []
        if args.offsets is None:
            args.offsets = []
        else:
            if len(args.offsets) != len(args.model):
                raise RuntimeError("Please specify one offset per model")
        for fn, off in zip_longest(args.model, args.offsets, fillvalue=None):
            if fn in ["human", "sawtooth"]:
                label = None
                p = getattr(util, fn)
                d = {k: p[k] for k in "abs"}
                d['N0'] = p['N0']
            elif not os.path.exists(fn):
                sys.exit("File not found: %s" % fn)
            else:
                res = json.load(open(fn, "rt"))
                if args.step_function:
                    mod = res['model']
                    klass = getattr(model, mod['class'])
                    m = klass.from_dict(mod)
                    a = m.stepwise_values().astype('float')
                    s = m.s
                    d = {'a': m.stepwise_values(), 's': m.s, 'N0': res['N0']}
                else:
                    d = res
                label = res['model']['pid']
            d['g'] = args.g
            psfs.append((label, d, off or 0))
        # if args.median:
        #     dmed = {'s': psfs[-1][1]['s'], 'N0': psfs[-1][1]['N0']}
        #     for x in "ab":
        #         dmed[x] = np.median([p[1][x] for p in psfs], axis=0)
        #     psfs.append((None, dmed, 0))
        fig, series = plotting.plot_psfs(psfs, xlim=args.xlim, ylim=args.ylim,
                                         xlabel="Generations" if args.g is None else "Years",
                                         logy=args.logy)
        fig.savefig(args.out, bbox_inches='tight')
        if args.csv:
            with open(os.path.splitext(args.pdf)[0] + ".csv", "wt") as out:
                csv.writer(out).writerows(series)
