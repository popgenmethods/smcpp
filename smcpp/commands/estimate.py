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
        command.add_pop_parameters(parser)
        model = command.add_model_parameters(parser)
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
