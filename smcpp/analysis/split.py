import numpy as np
import json
import sys

from .. import logging
from ..model import SMCTwoPopulationModel
from smcpp.optimize.optimizers import TwoPopulationOptimizer
from smcpp.optimize.plugins import parameter_optimizer
import smcpp.defaults
from . import base

logger = logging.getLogger(__name__)


class SplitAnalysis(base.BaseAnalysis):

    def __init__(self, files, args):
        base.BaseAnalysis.__init__(self, files, args)
        assert self.npop == 2
        self._init_model(args.pop1, args.pop2)
        # Further initialization
        hs = {k: np.array([0., np.inf]) for k in self._hidden_states}
        self._init_inference_manager(args.polarization_error, hs)
        self._init_optimizer(
            args.outdir, args.algorithm, args.xtol, args.ftol, single=False
        )
        self._niter = 1

    def _validate_data(self):
        base.BaseAnalysis._validate_data(self)
        if not any(c.npop == 2 for c in self.contigs):
            logger.error(
                "Data contains no joint frequency spectrum "
                "information. Split estimation is impossible."
            )
            sys.exit(1)

    _OPTIMIZER_CLS = TwoPopulationOptimizer

    def _init_optimizer(self, outdir, algorithm, xtol, ftol, single):
        super()._init_optimizer(outdir, algorithm, xtol, ftol, single)
        self._optimizer.register_plugin(
            parameter_optimizer.ParameterOptimizer(
                "split", (0., self._max_split), "model"
            )
        )

    def _init_model(self, pop1, pop2):
        d = json.load(open(pop1, "rt"))
        self._theta = d["theta"]
        self._rho = d["rho"]
        self._hidden_states = d["hidden_states"]
        m1 = base._model_cls_d[d["model"]["class"]].from_dict(d["model"])
        d = json.load(open(pop2, "rt"))
        m2 = base._model_cls_d[d["model"]["class"]].from_dict(d["model"])
        self._hidden_states.update(d["hidden_states"])
        assert d["theta"] == self._theta
        self._max_split = m2._knots[-(len(smcpp.defaults.additional_knots) + 1)]
        self._model = SMCTwoPopulationModel(m1, m2, self._max_split * 0.5)
