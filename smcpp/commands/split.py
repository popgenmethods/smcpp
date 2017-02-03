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
import json

# Package imports
from ..logging import getLogger, add_debug_log
from ..analysis import SplitAnalysis
from . import command

logger = getLogger(__name__)
np.set_printoptions(linewidth=120, suppress=True)


class Split(command.EstimationCommand, command.ConsoleCommand):
    'Estimate split time in two population model'

    def __init__(self, parser):
        command.EstimationCommand.__init__(self, parser)
        parser.add_argument('pop1', metavar="model1.final.json",
                            help="marginal fit for population 1")
        parser.add_argument('pop2', metavar="model2.final.json",
                            help="marginal fit for population 2")
        parser.add_argument('data', nargs="+",
                            help="data file(s) in SMC++ format")

    def main(self, args):
        # Initialize the logger
        # Do this before calling super().main() so that
        # any debugging output generated there gets logged
        add_debug_log(os.path.join(args.outdir, ".debug.txt"))

        command.EstimationCommand.main(self, args)
        # Create output directory
        try:
            os.makedirs(args.outdir)
        except OSError:
            pass  # directory exists

        # Save all the command line args and stuff
        logger.debug(sys.argv)
        logger.debug(args)

        # Fill in some of the population-genetic parameters from previous model run
        # TODO ensure that these params agree in both models?
        d = json.load(open(args.pop1, "rt"))
        args.N0 = d['N0']
        args.theta = d['theta']
        args.rho = None

        # Construct analysis
        analysis = SplitAnalysis(args.data, args)
        analysis.run()
