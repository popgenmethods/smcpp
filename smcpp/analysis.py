import numpy as np
import json
import sys

from . import estimation_tools, _smcpp, util, logging, spline, data_filter, beta_de
import smcpp.defaults
from .model import SMCModel, SMCTwoPopulationModel
from smcpp.optimize.optimizers import SMCPPOptimizer, TwoPopulationOptimizer
from smcpp.optimize.plugins import analysis_saver, parameter_optimizer

logger = logging.getLogger(__name__)

_model_cls_d = {cls.__name__: cls for cls in (SMCModel, SMCTwoPopulationModel)}

class BaseAnalysis:
    "Base class for analysis of population genetic data."
    def __init__(self, files, args):
        # Misc. parameter initialiations
        self._args = args
        if args.cores is not None:
            _smcpp.set_num_threads(args.cores)
        self._N0 = .5e-4 / args.mu  # .0001 = args.mu * 2 * N0
        self._theta = 2. * self._N0 * args.mu
        logger.info("theta: %f", self._theta)
        if args.r is not None:
            self._rho = 2 * self._N0 * args.r
        else:
            self._rho = self._theta
        assert np.all(np.isfinite([self._rho, self._theta]))
        logger.info("rho: %f", self._rho)
        self._penalty = 0.
        self._niter = args.em_iterations
        if args.unfold:
            args.polarization_error = 0.
            logger.warning("Using unfolded SFS. The user should verify "
                           "that the ancestral allele has been correctly "
                           "coded.")
        if args.polarization_error > 0.:
            logger.debug("Polarization error p=%f", args.polarization_error)
        smcpp.defaults.regularization_degree = args.regularization_degree
        logger.debug("regularization degree=%d", smcpp.defaults.regularization_degree)

        # Load data and apply transformations to normalize
        pipe = self._pipeline = data_filter.DataPipeline(files)
        pipe.add_filter(load_data=data_filter.LoadData())
        pipe.add_filter(data_filter.RecodeNonseg(cutoff=args.nonseg_cutoff))
        pipe.add_filter(data_filter.Compress())
        pipe.add_filter(data_filter.BreakLongSpans(cutoff=100000))
        pipe.add_filter(data_filter.DropSmallContigs(100000))
        pipe.add_filter(watterson=data_filter.Watterson())
        pipe.add_filter(mutation_counts=data_filter.CountMutations(w=int(2e-3 * self._N0 / self._rho)))


    @property
    def populations(self):
        return self._pipeline['load_data'].populations

    def _init_optimizer(self, outdir, algorithm, xtol, ftol, single):
        self._optimizer = self._OPTIMIZER_CLS(self, algorithm, xtol, ftol, single)
        if outdir:
            self._optimizer.register_plugin(analysis_saver.AnalysisSaver(outdir))

    def rescale(self, x):
        return x / (2. * self._N0)

    def __len__(self):
        return sum(len(c) for c in self.contigs)


    def _init_inference_manager(self, polarization_error, hs):
        ## Create inference object which will be used for all further calculations.
        logger.debug("Creating inference manager...")
        d = {}
        max_n = {}
        a = {}
        self._ims = {}
        for c in self.contigs:
            d.setdefault(c.pid, []).append(c)
            max_n.setdefault(c.pid, -1)
            max_n[c.pid] = np.maximum(max_n[c.pid], c.n)
            a.setdefault(c.pid, []).append(tuple(c.a))
        for pid in d:
            logger.debug("Creating inference manager for %s", pid)
            data = [c.data for c in d[pid]]
            if len(pid) == 1:
                im = _smcpp.PyOnePopInferenceManager(max_n[pid], data,
                        hs[pid[0]],
                        pid)
            else:
                assert len(pid) == 2
                s = set(a[pid])
                assert len(s) == 1
                im = _smcpp.PyTwoPopInferenceManager(*(max_n[pid]), *s.pop(), data,
                        hs[pid[0]],
                        pid)
            im.model = self._model
            im.theta = self._theta
            im.rho = self._rho
            im.alpha = self._alpha = 1
            im.polarization_error = polarization_error
            self._ims[pid] = im

    # @property
    # def _data(self):
    #     return [c.data for c in self.contigs]

    def run(self, niter=None):
        'Perform the analysis.'
        self._optimizer.run(niter or self._niter)

    def Q(self):
        'Value of Q() function in M-step.'
        qq = [self._ims[pop].Q(separate=True) for pop in self._ims]
        qr = self._penalty * self.model.regularizer()
        qq = np.sum(qq)
        ret = qq - qr
        logger.debug("reg: %s", util.format_ad(qr))
        logger.debug("Q:   %s", util.format_ad(ret))
        return ret

    def E_step(self):
        'Perform E-step.'
        logger.info('Running E-step')
        for pop in self._ims:
            self._ims[pop].E_step()
        logger.info('E-step completed')

    def loglik(self):
        'Log-likelihood of data after most recent E-step.'
        ll = 0
        for pop in self._ims:
            ll += self._ims[pop].loglik()
        return ll - self._penalty * float(self.model.regularizer())

    @property
    def model(self):
        return self._model

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, a):
        self._alpha = a
        for im in self._ims.values():
            im.alpha = a

    @property
    def rho(self):
        return self._rho

    @rho.setter
    def rho(self, r):
        self._rho = r
        for im in self._ims.values():
            im.rho = r

    @property
    def contigs(self):
        return list(self._pipeline.results())

    @property
    def npop(self):
        'The number of populations contained in this analysis.'
        return len(self.populations)

    def dump(self, filename):
        'Dump result of this analysis to :filename:.'
        d = {'theta': self._theta, 'rho': self._rho, 'alpha': self._alpha}
        d['model'] = self.model.to_dict()
        d['hidden_states'] = {k: list(v) for k, v in self._hidden_states.items()}
        json.dump(d, open(filename + ".json", "wt"), sort_keys=True, indent=4)


class Analysis(BaseAnalysis):
    '''A dataset, model and inference manager to be used for estimation.'''
    def __init__(self, files, args):
        super().__init__(files, args)

        pipe = self._pipeline
        pipe.add_filter(data_filter.Thin(thinning=args.thinning))
        pipe.add_filter(data_filter.BinObservations(w=args.w))
        pipe.add_filter(data_filter.RecodeMonomorphic())
        pipe.add_filter(data_filter.Compress())
        pipe.add_filter(data_filter.Validate())
        pipe.add_filter(data_filter.Summarize())

        if self.npop != 1:
            logger.error("Please use 'smc++ split' to estimate two-population models")
            sys.exit(1)

        NeN0 = self._pipeline['watterson'].theta_hat / (2. * args.mu * self._N0)

        # Optionally initialize from pre-specified model
        if args.initial_model:
            d = json.load(open(args.initial_model, "rt"))
            self._theta = d['theta']
            self._rho = d['rho']
            self._model = _model_cls_d[d['model']['class']].from_dict(d['model'])
            hs = self.rescale(smcpp.estimation_tools.balance_hidden_states(self._model, args.knots))
            self._hidden_states = {k: hs for k in self.populations}
            self._knots = hs[1:-1]
            logger.debug("rebalanced hidden states: %s", self._hidden_states)
        else:
            if args.timepoints == "h":
                mc = self._pipeline['mutation_counts'].counts
                w = self._pipeline['mutation_counts'].w
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
                             learn_rho=True, single=not args.multi)
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


class SplitAnalysis(BaseAnalysis):
    def __init__(self, files, args):
        BaseAnalysis.__init__(self, files, args)
        assert self.npop == 2
        self._init_model(args.pop1, args.pop2)
        # Further initialization
        hs = {k: np.array([0., np.inf]) for k in self._hidden_states}
        self._init_inference_manager(args.polarization_error, hs)
        self._init_optimizer(args.outdir, args.algorithm, args.xtol, args.ftol, single=False)
        self._niter = 1

    def _validate_data(self):
        BaseAnalysis._validate_data(self)
        if not any(c.npop == 2 for c in self.contigs):
            logger.error("Data contains no joint frequency spectrum "
                         "information. Split estimation is impossible.")
            sys.exit(1)

    _OPTIMIZER_CLS = TwoPopulationOptimizer

    def _init_optimizer(self, outdir, algorithm, xtol, ftol, single):
        super()._init_optimizer(outdir, algorithm, xtol, ftol, single)
        self._optimizer.register_plugin(parameter_optimizer.ParameterOptimizer("split",
                                                   (0., self._max_split),
                                                   "model"))

    def _init_model(self, pop1, pop2):
        d = json.load(open(pop1, "rt"))
        self._theta = d['theta']
        self._rho = d['rho']
        self._hidden_states = d['hidden_states']
        m1 = _model_cls_d[d['model']['class']].from_dict(d['model'])
        d = json.load(open(pop2, "rt"))
        m2 = _model_cls_d[d['model']['class']].from_dict(d['model'])
        self._hidden_states.update(d['hidden_states'])
        assert d['theta'] == self._theta
        self._max_split = m2._knots[-(len(smcpp.defaults.additional_knots) + 1)]
        self._model = SMCTwoPopulationModel(m1, m2, self._max_split * 0.5)
