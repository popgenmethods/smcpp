import json
import numpy as np
import scipy.stats.mstats
import sklearn.mixture
import sys

from .. import estimation_tools, _smcpp, util, spline, data_filter, beta_de
from ..model import SMCModel
from . import base
import smcpp.defaults
from smcpp.optimize.optimizers import SMCPPOptimizer
from smcpp.optimize.plugins import analysis_saver, parameter_optimizer

import logging
logger = logging.getLogger(__name__)


class Analysis(base.BaseAnalysis):
    """A dataset, model and inference manager to be used for estimation."""

    def __init__(self, files, args):
        super().__init__(files, args)

        if self.npop != 1:
            logger.error("Please use 'smc++ split' to estimate two-population models")
            sys.exit(1)
        self._theta = self._pipeline["watterson"].theta_hat 
        self._N0 = self._theta / 4 / args.mu
        if args.r is not None:
            self._rho = 2 * self._N0 * args.r
        else:
            self._rho = self._theta

        logger.info("theta: %f", self._theta)
        logger.info("rho: %f", self._rho)

        if args.timepoints is not None:
            # allow for directly setting timepoints
            tp = np.array(args.timepoints) / 2 / self._N0
            assert (tp >= 0.).all(), 'timepoints should be non-negative'
            if np.isclose(tp[0], 0.):
                self.hidden_states = tp
                self._knots = tp[1:]
            else:
                self._knots = tp
                self.hidden_states = np.insert(tp, 0, 0.)
        else:
            self._knots = np.geomspace(1e1, 1e5, args.knots) / 2 / self._N0
            self.hidden_states = np.r_[0., self._knots]

        pipe = self._pipeline
        pipe.add_filter(data_filter.Thin(thinning=args.thinning))
        pipe.add_filter(data_filter.BinObservations(w=args.w))
        pipe.add_filter(data_filter.RecodeMonomorphic())
        pipe.add_filter(data_filter.Compress())
        pipe.add_filter(data_filter.Validate())
        pipe.add_filter(data_filter.DropUninformativeContigs())
        pipe.add_filter(data_filter.Summarize())

        self._init_model(args.spline)
        self._model[:] = np.log(self._model(self._knots))
        self._init_inference_manager(args.polarization_error, self.hidden_states)
        self.alpha = args.w
        self._init_optimizer(
            args.outdir,
            args.base,
            args.algorithm,
            args.xtol,
            args.ftol,
            learn_rho=args.r is None,
            single=not args.multi,
        )
        self._init_regularization(args)

    def _init_model(self, spline_class):
        ## Initialize model
        logger.debug("knots in coalescent scaling:\n%s", str(self._knots))
        logger.debug("hidden states in coalescent scaling:\n%s", str(self.hidden_states))
        spline_class = {
            "cubic": spline.CubicSpline,
            "bspline": spline.BSpline,
            "akima": spline.AkimaSpline,
            "pchip": spline.PChipSpline,
            "piecewise": spline.Piecewise,
        }[spline_class]
        assert self.npop == 1
        self._model = SMCModel(self._knots, self._N0, spline_class, self.populations[0])

    def _init_regularization(self, args):
        if self._args.lambda_:
            self._penalty = args.lambda_
        else:
            self._penalty = abs(self.Q()) * (10 ** -args.regularization_penalty)
        logger.debug("Regularization penalty: lambda=%g", self._penalty)

    _OPTIMIZER_CLS = SMCPPOptimizer

    def _init_optimizer(self, outdir, base, algorithm, xtol, ftol, learn_rho, single):
        super()._init_optimizer(outdir, base, algorithm, xtol, ftol, single)
        if learn_rho:
            rho_bounds = lambda: (self._theta / 100, 100 * self._theta)
            self._optimizer.register_plugin(
                parameter_optimizer.ParameterOptimizer("rho", rho_bounds)
            )


