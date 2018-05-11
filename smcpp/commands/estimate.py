import argparse
import numpy as np
import scipy.optimize
import pprint
import sys
import itertools
import sys
import time
import os
import os.path

# Package imports
from ..logging import getLogger
from smcpp.analysis.analysis import Analysis
from . import command
import smcpp.defaults

logger = getLogger(__name__)
np.set_printoptions(linewidth=120, suppress=True)


class Estimate(command.EstimationCommand, command.ConsoleCommand):
    "Estimate size history for one population"


    def __init__(self, parser):
        command.EstimationCommand.__init__(self, parser)
        '''Configure parser and parse args.'''
        model = parser.add_argument_group('Model parameters')
        model.add_argument("--initial-model", help=argparse.SUPPRESS)
        model.add_argument('--timepoints', type=str, default="h",
                           help="starting and ending time points of model. "
                                "this can be either a comma separated list of two numbers `t1,tK`"
                                "indicating starting and ending generations, "
                                "a single value, indicating the starting time point, "
                                "or the special value 'h' "
                                "indicating that they should be determined based on the data using an "
                                "heuristic calculation.")
        model.add_argument('--knots', type=int,
                           default=smcpp.defaults.knots,
                           help="number of knots to use in internal representation")
        model.add_argument('--hs', type=int,
                           default=2,
                           help="ratio of (# hidden states) / (# knots). Must "
                                "be an integer >= 1. Larger values will consume more "
                                "memory and CPU but are potentially more accurate. ")
        model.add_argument('--spline',
                           choices=["cubic", "pchip", "piecewise"],
                           default=smcpp.defaults.spline,
                           help="type of model representation "
                                "(smooth spline or piecewise constant) to use")
        pop_params = parser.add_argument_group('Population-genetic parameters')
        pop_params.add_argument('mu', type=float,
                                help="mutation rate per base pair per generation")
        pop_params.add_argument('-r', type=float,
                                help="recombination rate per base pair per generation. "
                                     "default: estimate from data.")
        parser.add_argument('data', nargs="+",
                            help="data file(s) in SMC++ format")


    def validate_args(self, args):
        # perform some sanity checking on the args
        if not (1e-11 <= args.mu <= 1e-5):
            logger.warn(
                "The per-generation mutation rate is %g. Is this correct?" % args.mu)


    def main(self, args):
        command.EstimationCommand.main(self, args)
        # Perform some validation on the arguments
        self.validate_args(args)
        # Construct analysis
        analysis = Analysis(args.data, args)
        analysis.run()
