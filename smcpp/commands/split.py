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
from smcpp.analysis.split import SplitAnalysis
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
        command.EstimationCommand.main(self, args)
        # Fill in some of the population-genetic parameters from previous model run
        # TODO ensure that these params agree in both models?
        d = json.load(open(args.pop1, "rt"))
        args.mu = d['theta'] / (2. * d['model']['N0'])
        args.r = None

        # Construct analysis
        analysis = SplitAnalysis(args.data, args)
        analysis.run()
