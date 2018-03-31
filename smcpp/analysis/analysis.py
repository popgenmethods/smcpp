import numpy as np
import json
import sys

from .. import estimation_tools, _smcpp, util, logging, spline, data_filter, beta_de
from ..model import SMCModel
from . import base
import smcpp.defaults
from smcpp.optimize.optimizers import SMCPPOptimizer
from smcpp.optimize.plugins import analysis_saver, parameter_optimizer

logger = logging.getLogger(__name__)

class Analysis(base.BaseAnalysis):
    '''A dataset, model and inference manager to be used for estimation.'''
    def __init__(self, files, args):
        super().__init__(files, args)

        pipe = self._pipeline
        pipe.add_filter(data_filter.Thin(thinning=args.thinning))
        pipe.add_filter(data_filter.BinObservations(w=args.w))
        pipe.add_filter(data_filter.RecodeMonomorphic())
        pipe.add_filter(data_filter.Compress())
        pipe.add_filter(data_filter.Validate())
        pipe.add_filter(data_filter.DropUninformativeContigs())
        pipe.add_filter(data_filter.Summarize())

        if self.npop != 1:
            logger.error("Please use 'smc++ split' to estimate two-population models")
            sys.exit(1)

        NeN0 = self._pipeline['watterson'].theta_hat / (2. * args.mu * self._N0)

        # Optionally initialize from pre-specified model
        if args.initial_model:
            d = json.load(open(args.initial_model, "rt"))
            logger.debug("Import model:\n%s", d)
            self._theta = d['theta']
            self._rho = d['rho']
            self._model = base._model_cls_d[d['model']['class']].from_dict(d['model'])
            hs = self.rescale(smcpp.estimation_tools.balance_hidden_states(self._model, len(self._model) + 2))
            self._hidden_states = {k: hs for k in self.populations}
            self._knots = hs[1:-1]
            logger.debug("rebalanced hidden states: %s", self._hidden_states)
        else:
            if args.timepoints == "h":
                mc = self._pipeline['mutation_counts'].counts
                w = self._pipeline['mutation_counts'].w
                if np.all(mc == 0):
                    logger.error("Heuristic used to calculate time points has failed, "
                                 "possibly due to having a lot of missing data. Please "
                                 "set the --timepoints option manually.")
                    raise RuntimeError()
                q = smcpp.beta_de.quantile(mc / w, 1. / w, np.geomspace(1e-2, .99, args.knots))
                tau = q / (2. * self._theta)
                self._knots = tau
                n = np.average([c.n[0] for c in self.contigs], 
                        weights=[len(c) for c in self.contigs])
                if n > 0:
                    t1 = 2. * NeN0 * self._N0 * -np.log(.001) / (n * (n - 1) / 2)
                    logger.info("calculated t1: %f gens", t1)
                    rt1 = self.rescale(t1)
                    if rt1 < self._knots[0]:
                        self._knots = np.r_[np.geomspace(rt1, self._knots[0], args.knots // 4, False), self._knots]
                logger.debug("Determined knots heuristically to be: %s", self._knots)
            else:
                try:
                    t1, tK = [float(x) / 2. / self._N0 for x in args.timepoints.split(",")]
                    self._knots = np.geomspace(t1, tK, args.knots, False)
                    logger.debug("Knots are: %s", self._knots)
                except:
                    raise RuntimeError("Could not parse time points. "
                                       "See documentation for --timepoints option.")
            hs = np.r_[0., self._knots, np.inf]
            self._hidden_states = {k: hs for k in self.populations}

        logger.debug("hidden states in coalescent scaling: %s", hs)

        self._init_model(args.spline)
        self._init_inference_manager(args.polarization_error, self._hidden_states)
        self.alpha = args.w
        self._model[:] = np.log(NeN0)
        self._model.randomize()
        self._init_optimizer(args.outdir,
                             args.algorithm, args.xtol, args.ftol,
                             learn_rho=args.r is None,
                             single=args.no_multi)
        self._init_regularization(args)


    def _init_model(self, spline_class):
        ## Initialize model
        logger.debug("knots in coalescent scaling:\n%s", str(self._knots))
        spline_class = {"cubic": spline.CubicSpline,
                        "bspline": spline.BSpline,
                        "akima": spline.AkimaSpline,
                        "pchip": spline.PChipSpline,
                        "piecewise": spline.Piecewise}[spline_class]
        assert self.npop == 1
        self._model = SMCModel(
            self._knots, self._N0,
            spline_class, self.populations[0])

    def _init_regularization(self, args):
        if self._args.lambda_:
            self._penalty = args.lambda_
        else:
            self._penalty = abs(self.Q()) * (10 ** -args.regularization_penalty)
        logger.debug("Regularization penalty: lambda=%g", self._penalty)

    _OPTIMIZER_CLS = SMCPPOptimizer

    def _init_optimizer(self, outdir, algorithm, xtol, ftol, learn_rho, single):
        super()._init_optimizer(outdir, algorithm, xtol, ftol, single)
        if learn_rho:
            rho_bounds = lambda: (self._theta / 100, 100 * self._theta)
            self._optimizer.register_plugin(
                    parameter_optimizer.ParameterOptimizer("rho", rho_bounds))


