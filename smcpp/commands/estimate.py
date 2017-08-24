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
from ..analysis import Analysis
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
        model.add_argument('--t1', type=float, 
                           default=smcpp.defaults.t1,
                           help="starting point of first piece, in generations")
        model.add_argument('--tK', type=float,
                           help="end-point of last piece, in generations",
                           default=smcpp.defaults.tK)
        model.add_argument('--knots', type=int,
                           default=smcpp.defaults.knots,
                           help="number of knots to use in internal representation")
        model.add_argument('--spline',
                           choices=["cubic", "pchip"],
                           default=smcpp.defaults.spline,
                           help="type of spline representation to use in model")
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
