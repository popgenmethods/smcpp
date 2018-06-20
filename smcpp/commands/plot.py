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


class Plot(command.Command, command.ConsoleCommand):
    "Plot size history from fitted model"

    def __init__(self, parser):
        command.Command.__init__(self, parser)
        parser.add_argument(
            "-g",
            type=float,
            help="Plot x-axis in years assuming generation time(s) of g",
        )
        parser.add_argument(
            "-s",
            "--step-function",
            action="store_true",
            help="Plot the step function used to represent the spline "
            "instead of spline itself.",
        )
        parser.add_argument(
            "--csv",
            "-c",
            action="store_true",
            help="Also output <plot.csv> containing the plotted points",
        )
        parser.add_argument(
            "--linear", action="store_true", help="plot y on linear axis"
        )
        parser.add_argument(
            "--knots", "-k", action="store_true", help="also plot spline knots"
        )
        parser.add_argument(
            "-t",
            "--offsets",
            type=float,
            nargs="+",
            help="list of offsets, one for each <model>, to shift x axes. "
            "Mainly useful for plotting aDNA",
        )
        parser.add_argument(
            "--stat", action="append", default=[], help=argparse.SUPPRESS
        )
        parser.add_argument("--mean", action="store_true", help=argparse.SUPPRESS)
        parser.add_argument(
            "-x", "--xlim", type=float, nargs=2, default=None, help="x-axis limits"
        )
        parser.add_argument(
            "-y", "--ylim", type=float, nargs=2, default=None, help="y-axis limits"
        )
        # parser.add_argument("-l", "--labels", type=str,
        # help="label for each plotted function", nargs="+")
        parser.add_argument(
            "out", type=str, help="output image", metavar="plot.(pdf|png|gif|jpeg)"
        )
        parser.add_argument("model", type=str, help="SMC++ models to plot", nargs="+")

    def main(self, args):
        command.Command.main(self, args)
        psfs = []
        if args.offsets is None:
            args.offsets = []
        else:
            if len(args.offsets) != len(args.model):
                raise RuntimeError("Please specify one offset per model")
        for fn, off in zip_longest(args.model, args.offsets, fillvalue=None):
            if not os.path.exists(fn):
                sys.exit("File not found: %s" % fn)
            res = json.load(open(fn, "rt"))
            if args.step_function:
                mod = res["model"]
                klass = getattr(model, mod["class"])
                m = klass.from_dict(mod)
                a = m.stepwise_values().astype("float")
                s = m.s
                d = {"a": m.stepwise_values(), "s": m.s, "N0": mod["N0"]}
            else:
                d = res
            d["g"] = args.g
            psfs.append((d, off or 0))
        fig, series = plotting.plot_psfs(
            psfs,
            xlim=args.xlim,
            ylim=args.ylim,
            xlabel="Generations" if args.g is None else "Years",
            knots=args.knots,
            logy=not args.linear,
            stats={s: getattr(np, s) for s in args.stat},
        )
        fig.savefig(args.out, bbox_inches="tight")
        if args.csv:
            with open(os.path.splitext(args.out)[0] + ".csv", "wt") as out:
                csv.writer(out).writerows(series)
