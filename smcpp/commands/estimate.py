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
from ..analysis import Analysis

logger = getLogger(__name__)
np.set_printoptions(linewidth=120, suppress=True)


def add_common_estimation_args(parser):
    data = parser.add_argument_group('data parameters')
    data.add_argument('--nonseg-cutoff', '-c',
                      help="recode nonsegregating spans > cutoff as missing. "
                      "default: do not recode.",
                      type=int)
    data.add_argument('--length-cutoff',
                      help="omit sequences < cutoff. default: 10000", default=10000, type=int)
    data.add_argument('--thinning', help="only emit full SFS every <k>th site. default: 400 * n.",
                      default=None, type=int, metavar="k")
    data.add_argument('--no-filter', help="do not drop contigs with extreme heterozygosity. "
                                          "(not recommended unless data set is small)",
                      action="store_true", default=False)
    # data.add_argument("--folded", action="store_true", default=False,
    #                         help="use folded SFS for emission probabilites. "
    #                              "useful if polarization is not known.")

    optimizer = parser.add_argument_group("Optimization parameters")
    optimizer.add_argument("--no-initialize", action="store_true", default=False, help=argparse.SUPPRESS)
    optimizer.add_argument('--em-iterations', type=int,
                           help="number of EM steps to perform", default=10)
    optimizer.add_argument('--algorithm', choices=["L-BFGS-B", "TNC"],
                           default="L-BFGS-B", help=argparse.SUPPRESS)
    optimizer.add_argument('--block-size', type=int, default=3,
                           help="number of blocks to optimize at a time for coordinate ascent")
    optimizer.add_argument('--factr', type=float, default=1e-9, help=argparse.SUPPRESS)
    optimizer.add_argument('--regularization-penalty', "-p",
                           type=float, help="regularization penalty", default=1.)
    optimizer.add_argument("--tolerance", type=float, default=1e-4,
                           help="stopping criterion for relative improvement in loglik "
                           "in EM algorithm")
    optimizer.add_argument('--Nmin', type=float,
                           help="Lower bound on effective population size (in units of N0)",
                           default=.01)

    hmm = parser.add_argument_group("HMM parameters")
    hmm.add_argument(
        '--M', type=int, help="number of hidden states", default=32)

    parser.add_argument("-o", "--outdir", help="output directory", default=".")
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help="increase debugging output, specify multiply times for more")


def init_parser(parser):
    '''Configure parser and parse args.'''
    # Add in parameters which are shared with the split command
    add_common_estimation_args(parser)

    model = parser.add_argument_group('Model parameters')
    model.add_argument('--pieces', type=str,
                       help="span of model pieces", default="32*1")
    model.add_argument('--t1', type=float,
                       help="starting point of first piece, in generations", default=100)
    model.add_argument('--tK', type=float, help="end-point of last piece, in generations",
                       default=40000.)
    model.add_argument('--knots', type=str, default="10",
                       help="number of knots to use in internal representation")
    # model.add_argument('--exponential-pieces', type=int,
    # nargs="+", help="piece(s) which have exponential growth")
    model.add_argument(
        "--prior-model", help="prior on model, i.e. result of previous SMC++ run")
    model.add_argument('--offset', type=float, default=0.,
                       help="offset (in coalescent units) to use "
                            "when calculating time points")
    model.add_argument('--spline',
                       choices=["cubic", "akima", "pchip"],
                       default="cubic", help="type of spline representation to use in model")

    pop_params = parser.add_argument_group('Population-genetic parameters')
    pop_params.add_argument('--N0', default=1e4, type=float,
                            help="reference effective (diploid) population size to scale output.")
    pop_params.add_argument('--theta', '-t', type=float,
                            help="population-scaled mutation rate. default: Watterson's estimator.")
    pop_params.add_argument('--rho', '-r', type=float,
                            help="fix recombination rate to this value. default: estimate from data.")
    parser.add_argument('data', nargs="+", help="data file(s) in SMC++ format")
                        


def validate_args(args):
    # perform some sanity checking on the args
    if args.theta is not None:
        pgm = args.theta / (2. * args.N0)
        if not (1e-12 <= pgm <= 1e-2):
            logger.warn(
                "The per-generation mutation rate is calculated to be %g. Is this correct?" % pgm)


def main(args):
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
