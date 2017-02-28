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

logger = getLogger(__name__)
np.set_printoptions(linewidth=120, suppress=True)


class Estimate(command.EstimationCommand, command.ConsoleCommand):
    "Estimate size history for one population"

    def __init__(self, parser):
        command.EstimationCommand.__init__(self, parser)
        '''Configure parser and parse args.'''
        model = parser.add_argument_group('Model parameters')
        model.add_argument('--pieces', type=str,
                           help="span of model pieces", default="4+15*3+4")
        model.add_argument('--t1', type=float, default=2e2,
                           help="starting point of first piece, in generations")
        model.add_argument('--tK', type=float, help="end-point of last piece, in generations")
        model.add_argument('--knots', type=int, default="7",
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
        parser.add_argument('data', nargs="+",
                            help="data file(s) in SMC++ format")

    def validate_args(self, args):
        # perform some sanity checking on the args
        if args.theta is not None:
            pgm = args.theta / (2. * args.N0)
            if not (1e-12 <= pgm <= 1e-2):
                logger.warn(
                    "The per-generation mutation rate is calculated to be %g. Is this correct?" % pgm)

    def main(self, args):
        command.EstimationCommand.main(self, args)
        # Perform some validation on the arguments
        self.validate_args(args)
        # Construct analysis
        analysis = Analysis(args.data, args)
        analysis.run()
