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
from ..analysis import SplitAnalysis

logger = getLogger(__name__)
np.set_printoptions(linewidth=120, suppress=True)

def init_parser(parser):
    '''Configure parser and parse args.'''
    # FIXME: argument groups not supported in subparsers
    pop_params = parser  # parser.add_argument_group('population parameters')
    model = parser  # parser.add_argument_group('model')
    hmm = parser  # parser.add_argument_group('HMM and fitting parameters')
    optimizer = parser
    hmm.add_argument('--thinning', help="emit full SFS every <k>th site",
                     default=None, type=int, metavar="k")
    hmm.add_argument('--M', 
                     type=int, help="number of hidden states", default=32)
    hmm.add_argument('--em-iterations', type=int,
                     help="number of EM steps to perform", default=10)
    optimizer.add_argument('--algorithm', choices=["L-BFGS-B", "TNC"],
            default="L-BFGS-B", help=argparse.SUPPRESS)
    optimizer.add_argument("--tolerance", type=float, default=1e-4,
            help="stopping criterion for relative improvement in loglik "
                 "in EM algorithm")
    optimizer.add_argument('--Nmin', type=float,
                           help="Lower bound on effective population size (in units of N0)",
                           default=.01)
    optimizer.add_argument('--regularization-penalty', "-p",
                           type=float, help="regularization penalty", default=1.)
    pop_params.add_argument('--N0', default=1e4, type=float,
                            help="reference effective (diploid) population size to scale output.")
    hmm.add_argument('--length-cutoff',
                     help="omit sequences < cutoff", default=0, type=int)
    parser.add_argument("-o", "--outdir", help="output directory",
                        default=".", widget="DirChooser")
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help="increase debugging output, specify multiply times for more")
    parser.add_argument('pop1', help="marginal fit for population 1", widget="MultiFileChooser")
    parser.add_argument('pop2', help="marginal fit for population 2", widget="MultiFileChooser")
    parser.add_argument('data', nargs="+", help="data file(s) in SMC++ format", widget="MultiFileChooser")


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

    ## Construct analysis
    analysis = SplitAnalysis(args.data, args)
    analysis.run()
