# Base class; subclasses will automatically show up as subcommands
import numpy as np
import argparse
import os
import os.path
import sys
import logging

from .. import _smcpp
from ..log import setup_logging, add_debug_log
import smcpp.defaults

logger = logging.getLogger(__name__)

def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
         raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue

class ConsoleCommand:
    def __init__(self, parser):
        pass

class Command:
    def __init__(self, parser):
        '''Configure parser and parse args.'''
        parser.add_argument('-v', '--verbose', action='count', default=0,
                help="increase debugging output, specify multiply times for more")
        parser.add_argument('--seed', type=int, default=0, help=argparse.SUPPRESS)
        parser.add_argument('--cores', type=int, default=None, 
                help="Number of worker processes / threads "
                     "to use in parallel calculations")

    def main(self, args):
        np.random.seed(args.seed)
        setup_logging(args.verbose)
        smcpp.defaults.cores = args.cores

class EstimationCommand(Command):
    def __init__(self, parser):
        super().__init__(parser)
        add_common_estimation_args(parser)

    def main(self, args):
        if not os.path.isdir(args.outdir):
            os.makedirs(args.outdir)
        # Initialize the logger
        # Do this before calling super().main() so that
        # any debugging output generated there gets logged
        add_debug_log(os.path.join(args.outdir, ".debug.txt"))
        super().main(args)
        logger.debug(sys.argv)
        logger.debug(args)


def add_common_estimation_args(parser):
    parser.add_argument("-o", "--outdir", help="output directory", default=".")
    parser.add_argument("--base", help="base for file output. outputted files will have the form {base}.final.json, etc.", 
                        default="model")
    parser.add_argument('--timepoints', type=float, default=None, nargs=2,
                        help="start time of model (in generations)")
    data = parser.add_argument_group('data parameters')
    data.add_argument('--length-cutoff', help=argparse.SUPPRESS, type=int, default=None)
    data.add_argument('--nonseg-cutoff', '-c',
                      help="recode nonsegregating spans > cutoff as missing. "
                      "default: do not recode.",
                      type=int)
    data.add_argument('--thinning', help="only emit full SFS every <k>th site. (k > 0)",
                      default=None, type=check_positive, metavar="k")
    data.add_argument('-w', default=100, help="window size. sites are grouped into blocks of size <w>. "
                      "each block is coded as polymorphic or nonpolymorphic. "
                      "the default w=100 matches PSMC. setting w=1 performs no windowing "
                      "but may be more susceptible to model violations, in particular tracts "
                      " of hypermutated sites.", type=int)
    optimizer = parser.add_argument_group("Optimization parameters")
    optimizer.add_argument("--no-initialize", action="store_true", default=False, help=argparse.SUPPRESS)
    optimizer.add_argument('--em-iterations', type=int,
                           help="number of EM steps to perform", default=20)
    optimizer.add_argument('--algorithm',
                           choices=["Powell", "L-BFGS-B", "TNC"],
                           default="L-BFGS-B", help="optimization algorithm. Powell's method "
                                   "is used by {P,MS}MC and does not require gradients. It may "
                                   "be faster in some cases.")
    optimizer.add_argument('--multi', default=False, action="store_true",
                           help="update multiple blocks of coordinates at once")
    optimizer.add_argument("--ftol", type=float,
                           default=smcpp.defaults.ftol,
                           help="stopping criterion for relative improvement in loglik "
                           "in EM algorithm. algorithm will terminate when "
                           "|loglik' - loglik| / loglik < ftol")
    optimizer.add_argument('--xtol', type=float,
                           default=smcpp.defaults.xtol,
                           help=r"x tolerance for optimizer. "
                           "optimizer will stop when |x' - x|_\infty < xtol")
    optimizer.add_argument('--Nmax', type=float,
            default=smcpp.defaults.maximum_population_size,
            help="upper bound on scaled effective population size")
    optimizer.add_argument('--Nmin', type=float,
            default=smcpp.defaults.minimum_population_size,
            help="lower bound on scaled effective population size")
    optimizer.add_argument('--regularization-penalty', '-rp',
                           type=float, help="regularization penalty",
                           default=smcpp.defaults.regularization_penalty)
    optimizer.add_argument('--lambda', dest="lambda_", type=float, help=argparse.SUPPRESS)
    add_hmm_args(parser)

def add_hmm_args(parser):
    polarization = parser.add_mutually_exclusive_group(required=False)
    polarization.add_argument("--unfold", action="store_true", default=False,
                              help="use unfolded SFS (alias for -p 0.0)")
    polarization.add_argument('--polarization-error', '-p',
                              metavar='p', type=float, default=0.5,
                              help="uncertainty parameter for polarized SFS: observation (a,b) "
                              "has probability [(1-p)*CSFS_{a,b} + p*CSFS_{2-a,n-2-b}]. "
                              "default: 0.5")

def add_model_parameters(parser):
    model = parser.add_argument_group('Model parameters')
    # model.add_argument('--timepoints', type=str, default="h",
    #                    help="starting and ending time points of model. "
    #                         "this can be either a comma separated list of two numbers `t1,tK`"
    #                         "indicating starting and ending generations, "
    #                         "a single value, indicating the starting time point, "
    #                         "or the special value 'h' "
    #                         "indicating that they should be determined based on the data using an "
    #                         "heuristic calculation.")
    model.add_argument('--knots', type=int,
                       default=smcpp.defaults.knots,
                       help="number of knots to use in internal representation")
    # model.add_argument('--hs', type=int,
    #                    default=2,
    #                    help="ratio of (# hidden states) / (# knots). Must "
    #                         "be an integer >= 1. Larger values will consume more "
    #                         "memory and CPU but are potentially more accurate. ")
    model.add_argument('--spline',
                       choices=["cubic", "pchip", "piecewise"],
                       default=smcpp.defaults.spline,
                       help="type of model representation "
                            "(smooth spline or piecewise constant) to use")
    return model

def add_pop_parameters(parser):
    pop_params = parser.add_argument_group('Population-genetic parameters')
    pop_params.add_argument('mu', type=float,
                            help="mutation rate per base pair per generation")
    pop_params.add_argument('-r', type=float,
                            help="recombination rate per base pair per generation. "
                                 "default: estimate from data.")
    return pop_params
