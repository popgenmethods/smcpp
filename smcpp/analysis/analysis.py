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

        NeN0 = self._pipeline["watterson"].theta_hat / (2. * args.mu * self._N0)
        m = SMCModel([1.], self._N0, spline.Piecewise, None)
        m[:] = np.log(NeN0)
        hs = estimation_tools.balance_hidden_states(m, 2 + args.knots)
        if args.timepoints is not None:
            t1, tK = [x / 2 / self._N0 for x in args.timepoints]
        else:
            t1 = tK = None
        hs /= (2 * self._N0)
        self.hidden_states = hs
        self._init_knots(hs, t1, tK)

        self._init_model(args.spline)
        hs0 = hs
        self.hidden_states = [0., np.inf]
        self._init_inference_manager(args.polarization_error, self.hidden_states)
        self.alpha = 1
        self._model[:] = np.log(NeN0)
        self._model.randomize()
        self._init_optimizer(
            args.outdir,
            args.base,
            args.algorithm,
            args.xtol,
            args.ftol,
            learn_rho=False,
            single=False
        )
        self._init_regularization(args)
        self.run(1)

        pipe = self._pipeline
        pipe.add_filter(data_filter.Thin(thinning=args.thinning))
        pipe.add_filter(data_filter.BinObservations(w=args.w))
        pipe.add_filter(data_filter.RecodeMonomorphic())
        pipe.add_filter(data_filter.Compress())
        pipe.add_filter(data_filter.Validate())
        pipe.add_filter(data_filter.DropUninformativeContigs())
        pipe.add_filter(data_filter.Summarize())
        try:
            self._empirical_tmrca(2 * args.knots)
            hs = np.r_[0., self._etmrca_quantiles, np.inf]
        except Exception as e:
            logger.warn("Mixture model failed for setting hidden states. Error was: %s", e)
            hs = estimation_tools.balance_hidden_states(m, 2 * args.knots) / 2 / self._N0
        self.hidden_states = hs
        self._init_knots(hs, t1, tK)
        m = self._model
        self._init_model(args.spline)
        self._model[:] = np.log(m(self._knots))
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
        spline_class = {
            "cubic": spline.CubicSpline,
            "bspline": spline.BSpline,
            "akima": spline.AkimaSpline,
            "pchip": spline.PChipSpline,
            "piecewise": spline.Piecewise,
        }[spline_class]
        assert self.npop == 1
        self._model = SMCModel(self._knots, self._N0, spline_class, self.populations[0])

    def _init_knots(self, hs, t1, tK):
        self._knots = hs[1:-1:2]
        mult = np.mean(self._knots[1:] / self._knots[:-1])
        k0 = self._knots[0]
        t = t1 or k0
        a = []
        while t < k0:
            a = np.r_[a, t]
            t *= mult
        self._knots = np.r_[a, self._knots]
        if tK is not None and tK > self._knots[-1]:
            self._knots = np.r_[self._knots, tK]
        logger.debug("Knots are: %s", self._knots)

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


    def _empirical_tmrca(self, k):
        '''Calculate the empirical distribution of TMRCA in the
        distinguished lineages by counting mutations'''
        w = self._pipeline['mutation_counts'].w
        X = self._pipeline['mutation_counts'].counts
        logger.debug("Computing quantiles of TMRCA distribution from M=%d TMRCA samples", len(X))
        logger.debug("Unresampled quantiles (0/10/25/50/75/100): %s",
                     scipy.stats.mstats.mquantiles(X, [0, .1, .25, .5, .75, 1.]))
        # fit a k-poisson mixture model
        gmm = sklearn.mixture.GaussianMixture(n_components=k).fit(X[:, None])
        Y = gmm.sample(n_samples=100000)[0]
        p = np.logspace(np.log10(.01), np.log10(.99), k)
        q = scipy.stats.mstats.mquantiles(Y[Y>0], p) / (2 * self._theta * w)
        logger.debug("Quantiles: %s", " ".join("F(%g)=%g" % c for c in zip(q, p)))
        # 2 * E(TMRCA) * self._theta ~= q
        logger.debug("empirical TMRCA distribution: %s", q)
        self._etmrca_quantiles = q

