import numpy as np
import functools
import json
import sys
import os.path
import ad
import multiprocessing
import os
import scipy.stats
import concurrent.futures as futures
import itertools as it

from . import estimation_tools, _smcpp, util, logging, jcsfs, spline
import smcpp.defaults
from .contig import Contig
from .model import SMCModel, SMCTwoPopulationModel, PiecewiseModel
from smcpp.optimize.optimizers import SMCPPOptimizer, TwoPopulationOptimizer
from smcpp.optimize.plugins import analysis_saver, parameter_optimizer
import smcpp.beta_de

logger = logging.getLogger(__name__)

_model_cls_d = {cls.__name__: cls for cls in (SMCModel, SMCTwoPopulationModel)}

def thread_pool():
    cpu_count = os.cpu_count()
    return futures.ThreadPoolExecutor(cpu_count)

class BaseAnalysis:
    "Base class for analysis of population genetic data."
    def __init__(self, files, args):
        # Misc. parameter initialiations
        self._args = args
        self._N0 = .5e-4 / args.mu  # .0001 = args.mu * 2 * N0
        self._penalty = 0.
        self._niter = args.em_iterations
        args.solver_args = {}
        if args.unfold:
            args.polarization_error = 0.
            logger.warning("Using unfolded SFS. The user should verify "
                           "that the ancestral allele has been correctly "
                           "coded.")
        if args.polarization_error > 0.:
            logger.debug("Polarization error p=%f", args.polarization_error)
        smcpp.defaults.regularization_degree = args.regularization_degree
        logger.debug("regularization degree=%d", smcpp.defaults.regularization_degree)
        # Data-related stuff
        self._load_data(files)
        self._validate_data()
        self._filter_data()
        self._watterson()
        self._recode_nonseg(args.nonseg_cutoff)


    def _init_optimizer(self, outdir, algorithm, xtol, ftol, single):
        self._optimizer = self._OPTIMIZER_CLS(self, algorithm, xtol, ftol, single)
        if outdir:
            self._optimizer.register_plugin(analysis_saver.AnalysisSaver(outdir))

    def rescale(self, x):
        return x / (2. * self._N0)


    def __len__(self):
        return sum(len(c) for c in self._contigs)


    ## PRIVATE INIT FUNCTIONS
    def _load_data(self, files):
        ## Parse each data set into an array of observations
        logger.info("Loading data...")
        self._files = estimation_tools.files_from_command_line_args(files)
        self._contigs = estimation_tools.load_data(self._files)
        self._L = sum([d[:,0].sum() for d in self._data])
        logger.info("%.2f Gb of data", self._L * 1e-9)
        pops = set(c.pid for c in self._contigs)
        unique_pops = list({x for p in pops for x in p})
        assert len(unique_pops) <= 2, (
                "Only one or two populations are supported, but the "
                "following were found in the data: %r" % unique_pops)
        assert len(unique_pops) <= 2
        self._populations = tuple(unique_pops)
        for c in self._contigs:
            assert len(c.n) == len(c.a)
            assert c.a.max() <= 2
            assert c.a.min() >= 0
            assert c.a.sum() == 2
            assert c.data.shape[1] == 1 + 3 * len(c.n)
            logger.debug(c)
        logger.info("%d population%s", self.npop, "" if self.npop == 1 else "s")
        ns = self._ns = np.array([sum(c.n) for c in self._contigs])

    def _validate_data(self):
        for c in self._contigs:
            assert c.data.flags.c_contiguous
            bad = (np.all(c.data[:, 1::3] == c.a[None, :], axis=1) &
                   np.all(c.data[:, 2::3] == c.n[None, :], axis=1))
            if np.any(bad):
                logger.error("In file %s, observations %s:", c.fn, np.where(bad)[0])
                logger.error("Data set contains sites where every "
                        "individual is homozygous for the derived allele. "
                        "Please recode these as non-segregating (homozygous ancestral).")
                sys.exit(1)
            bad = (c.data[:, 0] <= 0 |
                   np.any(c.data[:, 1::3] > c.a[None, :], axis=1) |
                   np.any(c.data[:, 2::3] > c.data[:, 3::3], axis=1) |
                   np.any(c.data[:, 3::3] > c.n[None, :], axis=1))
            if np.any(bad):
                logger.error("File %s has invalid observations "
                             "(span <= 0 | a > 2 | b > n | n > sample size): %s",
                             c.fn, np.where(bad)[0])
                sys.exit(1)


    def _recode_nonseg(self, cutoff):
        self._contigs = estimation_tools.recode_nonseg(self._contigs, cutoff)

    def _realign(self, w):
        with thread_pool() as p:
            f = [p.submit(smcpp._smcpp.realign, c, w) for c in self._contigs]
            futures.wait(f, return_when=futures.FIRST_EXCEPTION)

    def _filter_data(self):
        ws = self._args.w or 100000
        if not len(self) > 10 * ws:
            logger.debug("Too little data to filter")
            return
        self._realign(ws)
        bincounts = [c.windowed_mutation_counts(ws) for c in self._contigs]
        mc = [m for _, muts, nmiss in bincounts 
                for m, nm in zip(muts, nmiss)
                if nm > .5 * ws]
        l, u = scipy.stats.mstats.mquantiles(mc, [.05, .95])
        logger.debug("data filtering: %d < # mutations < %d in %dbp windows", l, u, ws)
        for c, (bins, counts, _) in zip(self._contigs, bincounts):
            c.filter_windows(ws, bins, counts, l, u)


    def _watterson(self):
        self._watterson = estimation_tools.watterson_estimator(self._contigs)
        logger.debug("Watterson estimator: %f", self._watterson)


    def _perform_thinning(self, thinning):
        # thin each dataset
        npop = np.array([c.npop for c in self._contigs])
        if isinstance(thinning, int):
            thinning = np.array([thinning] * len(self._contigs))
        if thinning is None:
            thinning = (1000 * np.log(2 + self._ns)).astype("int")   # 500  * ns
        thinning[npop > 1] = 0
        if np.any(thinning > 1):
            logger.info("Thinning...")
            logger.debug("Thinning parameters: %s", thinning)
            new_data = estimation_tools.thin_dataset(self._data, thinning)
            self._contigs = [Contig(data=d, pid=c.pid, fn=c.fn, n=c.n, a=c.a) 
                             for c, d in zip(self._contigs, new_data)]
            logger.debug('Thinning complete')
        # elif np.any(ns > 0):
        #     logger.warn("Not thinning yet undistinguished lineages are present")


    def _empirical_tmrca(self, k, w):
        '''Calculate the empirical distribution of TMRCA in the
        distinguished lineages by counting mutations'''
        logger.debug("EMTRCA with w=%d", w)
        self._realign(w)
        wms = [c.windowed_mutation_counts(w, n2=True) for c in self._contigs] 
        Xw = [[m / nm, nm] for _, mut, nmiss in wms 
                             for m, nm in zip(mut, nmiss) 
                             if nm > .5 * w]
        if not Xw:
            Xw = [[2 * self._theta, w]] * 100
        X, w = np.transpose(Xw)
        # Beta density estimation
        N = int(1e4)
        M = len(X)
        h = 1 / M
        logger.debug("Computing quantiles of TMRCA distribution from M=%d TMRCA samples", M)
        logger.debug("Unresampled quantiles (0/10/25/50/75/100): %s",
                     scipy.stats.mstats.mquantiles(X, [0, .1, .25, .5, .75, 1.]))
        p = np.logspace(np.log10(.01), np.log10(.99), k)
        # with futures.ProcessPoolExecutor() as pool:
        q = smcpp.beta_de.quantile(X, h, p)
        logger.debug("Quantiles: %s", " ".join("F(%f)=%f" % c for c in zip(q, p)))
        # 2 * E(TMRCA) * self._theta ~= q
        q /= (2. * self._theta)
        logger.debug("empirical TMRCA distribution: %s", q)
        self._etmrca_quantiles = q


    def _compress(self):
        for c in self._contigs:
            c.data = estimation_tools.compress_repeated_obs(c.data)


    def _normalize_data(self, length_cutoff, filter):
        ## break up long spans
        self._contigs, attrs = estimation_tools.break_long_spans(self._contigs, length_cutoff)
        if not attrs:
            logger.error("No contigs remain after pre-processing! Check --length-cutoff.")
            sys.exit(1)
        w, het = np.array([a[2:] for k in attrs for a in attrs[k]]).T
        self._het = avg = np.average(het, weights=w)
        logger.debug("Heterozygosity: %f", self._het)
        if self._het == 0:
            logger.error("Data contain *no* mutations. Inference is impossible.")
            sys.exit(1)
        n = len(het)
        if n == 1:
            avg = 0.
            sd = np.inf
        else:
            var = np.average((het - avg) ** 2, weights=w) * (n / (n - 1.))
            sd = np.sqrt(var)
            logger.debug("Average/sd het:%f(%f)", avg, sd)
            if filter:
                logger.debug("Keeping contigs within +-3 s.d. of mean")
        logger.debug("Average heterozygosity (derived / total bases) by data set (* = dropped)")
        ci = 0
        tpl = "%15d%15d%15d%12g"
        new_contigs = []
        for fn, key in zip(self._files, attrs):
            logger.debug(fn + ":")
            for attr in attrs[key]:
                het = attr[-1]
                mytpl = tpl
                if not filter or abs(het - avg) <= 3 * sd:
                    new_contigs.append(self._contigs[ci])
                else:
                    mytpl += " *"
                logger.debug(mytpl % attr)
                ci += 1
        self._contigs = new_contigs

    def _calculate_t1_tK(self, args):
        Ne = self._watterson / self._theta
        logger.debug("Ne: %f", Ne)
        Ne *= 2 * self._N0
        n = 2 + max(self._ns)
        t1 = args.t1 or 200 + np.log(.9) / (-n * (n - 1) / 2) * Ne
        if t1 <= 0:
            logger.error("--t1 should be >0")
            sys.exit(1)
        logger.debug("setting t1=%f", t1)
        tK = args.tK or (2 + 1.16 ** 0.5) * Ne
        if tK <= 0:
            logger.error("--tK should be >0")
            sys.exit(1)
        logger.debug("setting tK=%f", tK)
        if tK <= t1:
            logger.error("tK <= t1? Possible weirdness in data")
            sys.exit(1)
        self._t1 = self.rescale(t1)
        self._tK = self.rescale(tK)


    def _init_inference_manager(self, polarization_error):
        ## Create inference object which will be used for all further calculations.
        logger.debug("Creating inference manager...")
        d = {}
        max_n = {}
        a = {}
        self._ims = {}
        for c in self._contigs:
            d.setdefault(c.pid, []).append(c)
            max_n.setdefault(c.pid, -1)
            max_n[c.pid] = np.maximum(max_n[c.pid], c.n)
            a.setdefault(c.pid, []).append(tuple(c.a))
        for pid in d:
            logger.debug("Creating inference manager for %s", pid)
            data = [c.data for c in d[pid]]
            if len(pid) == 1:
                im = _smcpp.PyOnePopInferenceManager(max_n[pid], data,
                        self._hidden_states[pid[0]],
                        pid, polarization_error)
            else:
                assert len(pid) == 2
                s = set(a[pid])
                assert len(s) == 1
                im = _smcpp.PyTwoPopInferenceManager(*(max_n[pid]), *s.pop(), data,
                                                     self._hidden_states[pid[0]],
                                                     pid,
                                                     polarization_error)
            im.model = self._model
            im.theta = self._theta
            im.rho = self._rho
            im.alpha = self._alpha = .01
            self._ims[pid] = im
        self._max_n = np.max(list(max_n.values()), axis=0)
        self._model.randomize()

    @property
    def _data(self):
        return [c.data for c in self._contigs]

    def run(self):
        'Perform the analysis.'
        self._optimizer.run(self._niter)

    def Q(self):
        'Value of Q() function in M-step.'
        qq = []
        with thread_pool() as executor:
            fs = []
            for na in self._ims:
                fs.append(executor.submit(self._ims[na].Q, separate=True))
            for x in futures.as_completed(fs):
                qq.append(x.result())
        qr = self._penalty * self.model.regularizer()
        qq = np.sum(qq)
        ret = qq - qr
        logger.debug("reg: %s", util.format_ad(qr))
        logger.debug("Q:   %s", util.format_ad(ret))
        return ret

    def E_step(self):
        'Perform E-step.'
        logger.info('Running E-step')
        with thread_pool() as executor:
            fs = []
            for na in self._ims:
                fs.append(executor.submit(self._ims[na].E_step))
            futures.wait(fs)
        logger.info('E-step completed')

    def loglik(self):
        'Log-likelihood of data after most recent E-step.'
        ll = 0
        with thread_pool() as executor:
            fs = []
            for na in self._ims:
                fs.append(executor.submit(self._ims[na].loglik))
            for x in futures.as_completed(fs):
                ll += x.result()
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
    def npop(self):
        'The number of populations contained in this analysis.'
        return len(self._populations)

    def dump(self, filename):
        'Dump result of this analysis to :filename:.'
        d = {'theta': self._theta, 'rho': self._rho}
        d['model'] = self.model.to_dict()
        d['hidden_states'] = {k: v.tolist() for k, v in self._hidden_states.items()}
        json.dump(d, open(filename + ".json", "wt"), sort_keys=True, indent=4)


class Analysis(BaseAnalysis):
    '''A dataset, model and inference manager to be used for estimation.'''
    def __init__(self, files, args):
        super().__init__(files, args)
        if self.npop != 1:
            logger.error("Please use 'smc++ split' to estimate two-population models")
            sys.exit(1)

        # Perform initial filtering for weird contigs
        self._normalize_data(args.length_cutoff, not args.no_filter)

        # Initialize members
        self._init_parameters(args.mu, args.r)

        # Estimate empirical TMRCA distribution for distinguished pairs
        self._empirical_tmrca(2 * args.knots,
                              args.w or .1 * self._cM)
        self._hidden_states = np.r_[0, self._etmrca_quantiles, np.inf]
        self._M = len(self._hidden_states)

        # Figure out knot placement.
        self._calculate_t1_tK(args)
        self._init_knots()
        self._init_model(self._N0, args.spline)

        # Optionally initialize from pre-specified model
        if args.initial_model:
            d = json.load(open(args.initial_model, "rt"))
            self._theta = d['theta']
            self._rho = d['rho']
            self._hidden_states = {k: np.array(v) for k, v in d['hidden_states'].items()}
            self._model = _model_cls_d[d['model']['class']].from_dict(d['model'])
        else:
            self._preinitialize()

        # Thin the data
        self._perform_thinning(args.thinning)
        self._compress()

        self._init_inference_manager(args.polarization_error)
        self._init_optimizer(args.outdir,
                             args.algorithm, args.xtol, args.ftol,
                             learn_rho=args.r is None, single=False)


    def _init_parameters(self, mu, r):
        ## Set theta and rho to their default parameters
        self._theta = 2. * self._N0 * mu
        logger.info("theta: %f", self._theta)
        if r is not None:
            self._rho = 2 * self._N0 * r
        else:
            self._rho = self._theta
        self._cM = 1e-2 / (self._rho / (2 * self._N0))
        assert np.all(np.isfinite([self._rho, self._theta]))
        logger.info("rho: %f", self._rho)


    def _init_knots(self):
        self._knots = self._hidden_states[1:-1:2]
        mult = np.mean(self._knots[1:] / self._knots[:-1])
        t = self._t1
        k0 = self._knots[0]
        a = []
        while t < k0:
            a = np.r_[a, t]
            t *= mult
        self._knots = np.r_[a, self._knots]
        if self._tK > self._knots[-1]:
            self._knots = np.r_[self._knots, self._tK]
        for x in smcpp.defaults.additional_knots:
            self._knots = np.r_[self._knots, x * self._knots[-1]]


    def _init_model(self, N0, spline_class):
        ## Initialize model
        logger.debug("knots in coalescent scaling:\n%s", str(self._knots))
        spline_class = {"cubic": spline.CubicSpline,
                        "bspline": spline.BSpline,
                        "akima": spline.AkimaSpline,
                        "pchip": spline.PChipSpline}[spline_class]
        self._y0 = y0 = self._het / (2. * self._theta)
        logger.debug("Long term avg. effective population size: %f", y0)
        y0 = np.log(y0)
        if self.npop == 1:
            self._model = SMCModel(
                self._knots, self._N0,
                spline_class, self._populations[0])
            self._model[:] = y0
        else:
            split = self.rescale(.5 * tK)  # just pick the midpoint as a starting value.
            mods = []
            for pid in self._populations:
                mods.append(SMCModel(self._knots, self._N0, spline_class, pid))
                mods[-1][:] = y0
            self._model = SMCTwoPopulationModel(mods[0], mods[1], split)

    def _preinitialize(self):
        args = self._args
        x = self.model[:]
        self._init_model(self._N0, args.spline)
        self.model[:] = x

        # Do a first pass to initialize regularizer and hidden states
        self._hidden_states = {k: [0.0, np.inf] for k in self._populations}
        self._init_inference_manager(args.polarization_error)
        self._init_optimizer(None, args.algorithm, args.xtol,
                             args.ftol, learn_rho=False, single=False)
        self.E_step()
        if self._args.lambda_:
            self._penalty = args.lambda_
        else:
            self._penalty = abs(self.Q()) * (10 ** -args.regularization_penalty)
        self._optimizer.run(1)
        logger.debug("Regularization penalty: lambda=%g", self._penalty)
        self._hidden_states = self.rescale(smcpp.estimation_tools.balance_hidden_states(
            self._model, 
            self._M
            ))[:-1]
        if args.t1 is None:  # set initial knot based on density of coalescence in rough demography
            self._t1 = max(self.rescale(50), smcpp.estimation_tools.calculate_t1(self._model, self._max_n, q=.5))
            logger.debug("Recomputed t1: %f", self._t1)
        if args.tK is None:
            self._tK = self._hidden_states[-2]
        self._init_knots()
        m0 = self._model
        self._init_model(self._N0, args.spline)
        self._model.match(m0)
        t12 = smcpp.estimation_tools.calculate_t1(self._model, 2, q=0.01)
        if t12 < self._hidden_states[1]:
            self._hidden_states[0] = t12
        else:
            self._hidden_states = self._hidden_states[1:]
        logger.debug("hidden states: %s", self._hidden_states)
        self._hidden_states = {k: self._hidden_states for k in self._populations}

    _OPTIMIZER_CLS = SMCPPOptimizer

    def _init_optimizer(self, outdir, algorithm, xtol, ftol, learn_rho, single):
        super()._init_optimizer(outdir, algorithm, xtol, ftol, single)
        # alpha_bounds = (.0, .2)
        # self._optimizer.register_plugin(
        #         parameter_optimizer.ParameterOptimizer("alpha", alpha_bounds))
        if learn_rho:
            rho_bounds = self._theta / 10, 10 * self._theta
            self._optimizer.register_plugin(
                    parameter_optimizer.ParameterOptimizer("rho", tuple(rho_bounds)))


class SplitAnalysis(BaseAnalysis):
    def __init__(self, files, args):
        BaseAnalysis.__init__(self, files, args)
        assert self.npop == 2
        self._init_model(args.pop1, args.pop2)
        self._normalize_data(args.length_cutoff, not args.no_filter)
        self._perform_thinning(args.thinning)
        # Further initialization
        self._init_inference_manager(args.polarization_error)
        self._init_optimizer(args.outdir, args.algorithm, args.xtol, args.ftol)
        self._niter = 1

    def _validate_data(self):
        BaseAnalysis._validate_data(self)
        if not any(c.npop == 2 for c in self._contigs):
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
        assert d['theta'] == self._theta
        self._max_split = m2._knots[-(len(smcpp.defaults.additional_knots) + 1)]
        self._model = SMCTwoPopulationModel(m1, m2, self._max_split * 0.5)
