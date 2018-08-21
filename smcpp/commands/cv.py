import argparse
import contextlib
import json
import numpy as np
import os
import os.path
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

from . import estimate, command
from ..logging import getLogger
from .. import model
from smcpp.analysis.analysis import Analysis

logger = getLogger(__name__)

@contextlib.contextmanager
def mark_completed(path):
    p = Path(path, ".done")
    yield p
    p.touch()


class Cv(command.EstimationCommand, command.ConsoleCommand):
    "Perform cross-validated estimation procedure."

    def __init__(self, parser):
        super().__init__(parser)
        model = command.add_model_parameters(parser)
        command.add_pop_parameters(parser)
        parser.add_argument("--initial-model", help=argparse.SUPPRESS)
        parser.add_argument(
            "--folds", type=int, default=2, help="number of folds for cross-validation"
        )
        parser.add_argument(
            "--fold", type=int, help="run a specific fold only, useful for parallelizing"
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
        if args.fold is not None and not (0 <= args.fold < args.folds):
            logger.error(
                "--fold should be an integer between 0 and --folds"
            )
            sys.exit(1)
        folds = np.array_split(np.arange(L), args.folds)
        basedir = args.outdir
        best_models = [None] * len(folds)
        def fold_path(i):
            return os.path.join(basedir, "fold{}".format(i))
        for i, fold in enumerate(folds):
            if args.fold is not None and args.fold != i:
                logger.debug("Skipping fold %d since '--fold %d' was specified", i, args.fold)
                continue
            fp = fold_path(i)
            with mark_completed(fp) as p:
                if p.exists():
                    logger.debug("Skipping fold %d (encountered %s)", i, p)
                    with open(os.path.join(fp, "model.best.json"), "rt") as f:
                        best_models[i] = model.SMCModel.from_dict(json.load(f)['model'])
                    continue
                args.outdir = fp
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
        if args.fold is not None:
            sys.exit(0)
        missing = [i for i in range(args.folds) 
                   if not Path(fold_path(i), ".done").exists()]
        if missing:
            logger.error("Not averaging models as the following folds have not been completed: %s", missing)
            sys.exit(0)
        logger.info("Averaging over folds")
        # STEP 2
        mavg = model.aggregate(*best_models)
        d = {"model": mavg.to_dict()}
        json.dump(
            d,
            open(os.path.join(basedir, "model.final.json"), "wt"),
            sort_keys=True,
            indent=4,
        )
