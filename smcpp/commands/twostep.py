import argparse
import json
import numpy as np
import os
import os.path
import shutil
import sys
from types import SimpleNamespace

from . import estimate, command
from ..logging import getLogger
from .. import model
from smcpp.analysis.analysis import Analysis

logger = getLogger(__name__)


class TwoStep(command.EstimationCommand, command.ConsoleCommand):
    "Perform two-step estimation procedure."

    def __init__(self, parser):
        super().__init__(parser)
        model = command.add_model_parameters(parser)
        command.add_pop_parameters(parser)
        parser.add_argument("--initial-model", help=argparse.SUPPRESS)
        parser.add_argument(
            "--folds", type=int, default=2, help="number of folds for cross-validation"
        )
        parser.add_argument("data", nargs="+", help="data file(s) in SMC++ format")

    def main(self, args):
        command.EstimationCommand.main(self, args)
        ## k-fold cv
        L = len(args.data)
        if not (2 <= args.folds <= L):
            logger.error(
                "--folds should be an integer between 2 and the number of contigs."
            )
            sys.exit(1)
        folds = np.array_split(np.arange(L), args.folds)
        basedir = args.outdir
        best_models = [None] * len(folds)
        for i, fold in enumerate(folds):
            args.outdir = os.path.join(basedir, "fold{}".format(i))
            os.makedirs(args.outdir, exist_ok=True)
            test = Analysis([args.data[j] for j in range(L) if j in fold], args)
            best = float("-Inf")
            for j in range(2, 10):
                args.regularization_penalty = j
                train = Analysis(
                    [args.data[k] for k in range(L) if k not in fold], args
                )
                train.run()
                test.model = train.model
                test.E_step()
                logger.debug(
                    "STEP 1a: rp=%d train=%f test=%f",
                    j,
                    float(train.loglik(True)),
                    float(test.loglik(False)),
                )
                if test.loglik(False) > best:
                    best_models[i] = train.model
                    shutil.copyfile(
                        os.path.join(args.outdir, "model.final.json"),
                        os.path.join(args.outdir, "model.best.json"),
                    )
        # STEP 2
        mavg = model.aggregate(*best_models)
        d = {"model": mavg.to_dict()}
        json.dump(
            d,
            open(os.path.join(basedir, "model.final.json"), "wt"),
            sort_keys=True,
            indent=4,
        )
