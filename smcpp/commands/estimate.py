'Fit SMC++ to data using the EM algorithm'
from __future__ import absolute_import, division, print_function
import argparse
import numpy as np
import scipy.optimize
import pprint
import sys
import itertools
import sys
import time
import os


# Package imports
from ..logging import getLogger, setup_logging

logger = getLogger(__name__)

from .. import _smcpp, util, estimation_tools
from ..model import SMCModel
from ..analysis import Analysis
from ..optimizer import SMCPPOptimizer

np.set_printoptions(linewidth=120, suppress=True)

def init_parser(parser):
    '''Configure parser and parse args.'''
    # FIXME: argument groups not supported in subparsers
    pop_params = parser  # parser.add_argument_group('population parameters')
    model = parser  # parser.add_argument_group('model')
    hmm = parser  # parser.add_argument_group('HMM and fitting parameters')
    optimizer = parser
    model.add_argument('--pieces', type=str,
                       help="span of model pieces", default="32*1")
    model.add_argument('--t1', type=float,
                       help="starting point of first piece, in generations", default=100)
    model.add_argument('--tK', type=float, help="end-point of last piece, in generations", 
                       default=40000.)
    model.add_argument('--knots', type=int, default=10, help="number of knots to use in internal representation")
    # model.add_argument('--exponential-pieces', type=int,
    #                    nargs="+", help="piece(s) which have exponential growth")
    model.add_argument("--initial-model", help="initial model, i.e. result of previous SMC++ run")
    hmm.add_argument('--thinning', help="emit full SFS every <k>th site",
                     default=None, type=int, metavar="k")
    hmm.add_argument('--M', 
                     type=int, help="number of hidden states", default=32)
    hmm.add_argument('--em-iterations', type=int,
                     help="number of EM steps to perform", default=10)
    optimizer.add_argument('--no-prefit', action="store_true",
                           help="skip model prefitting. (not recommended.)")
    optimizer.add_argument('--fixed-split', type=float,
                           help="instead of estimating split time, fix it to this value. (two-population models only.)")
    optimizer.add_argument("--fix-rho", default=False, action="store_true",
                           help="do not estimate recombination rate from data")
    optimizer.add_argument('--block-size', type=int, default=3,
                           help="number of blocks to optimizer at a time for coordinate ascent")
    optimizer.add_argument('--regularization-penalty',
                           type=float, help="regularization penalty", default=1.)
    optimizer.add_argument('--spline',
                           choices=["bspline", "cubic", "akima", "pchip"],
                           default="pchip", help="type of spline representation to use")
    optimizer.add_argument('--Nmin', type=float,
                           help="Lower bound on effective population size (in units of N0)",
                           default=.01)
    pop_params.add_argument('--N0', default=1e4, type=float,
                            help="reference effective (diploid) population size to scale output.")
    pop_params.add_argument('--theta', type=float,
                            help="population-scaled mutation rate. default: Watterson's estimator.")
    pop_params.add_argument('--rho', type=float,
                            help="population-scaled mutation rate. default: theta.")
    pop_params.add_argument("--folded", action="store_true", default=False,
                            help="use folded SFS for emission probabilites. useful if polarization is not known.")
    hmm.add_argument('--length-cutoff',
                     help="omit sequences < cutoff", default=0, type=int)
    parser.add_argument("--pdb", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("-o", "--outdir", help="output directory",
                        default=".", widget="DirChooser")
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help="increase debugging output, specify multiply times for more")
    parser.add_argument('data', nargs="+", help="data file(s) in SMC++ format", 
                        widget="MultiFileChooser")


def validate_args(args):
    # perform some sanity checking on the args
    if args.theta is not None:
        pgm = args.theta / (2. * args.N0)
        if not (1e-12 <= pgm <= 1e-2):
            logger.warn(
                "The per-generation mutation rate is calculated to be %g. Is this correct?" % pgm)


def main(args):
    if args.pdb:
        from pudb import set_trace
        sys.excepthook = lambda *args: set_trace()
    ## Create output directory
    try:
        os.makedirs(args.outdir)
    except OSError:
        pass  # directory exists

    ## Initialize the logger
    setup_logging(args.verbose, os.path.join(args.outdir, ".debug.txt"))

    ## Save all the command line args and stuff
    logger.debug(sys.argv)
    logger.debug(args)

    ## Perform some validation on the arguments
    validate_args(args)

    ## Construct analysis
    analysis = Analysis(args.data, args)
    analysis.run()
