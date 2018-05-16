import os.path
import sys

from types import SimpleNamespace
from . import estimate, command
from ..logging import getLogger
from smcpp.analysis.analysis import Analysis

logger = getLogger(__name__)

class TwoStep(command.EstimationCommand, command.ConsoleCommand):
    "Perform two-step estimation procedure."
    def __init__(self, parser):
        super().__init__(parser)
        model = command.add_model_parameters(parser)
        command.add_pop_parameters(parser)
        parser.add_argument('data', nargs="+",
                            help="data file(s) in SMC++ format")

    def main(self, args):
        command.EstimationCommand.main(self, args)
        print(args)
        s = SimpleNamespace(**vars(args))
        s.em_iterations = 1
        s.knots = 12
        s.timepoints = "h"
        s.initial_model = None
        print(s)
        ## STEP 1
        logger.info("STEP 1: Pre-estimation")
        analysis = Analysis(args.data, s)
        analysis.run()
        ## STEP 2
        logger.info("STEP 2: Estimation")
        args.initial_model = args.outdir + "/model.final.json"
        analysis = Analysis(args.data, args)
        analysis.run()
