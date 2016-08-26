import numpy as np
import functools
import json
import sys
import os.path
import scipy.optimize
import ad

from . import estimation_tools, _smcpp, util, logging, optimizer, jcsfs, spline
from .contig import Contig
from .model import SMCModel, SMCTwoPopulationModel

logger = logging.getLogger(__name__)


class Analysis:
    '''A dataset, model and inference manager to be used for estimation.'''
    def __init__(self, files, args):
        # Misc. parameter initialiations
        self._N0 = args.N0
        self._penalty = args.regularization_penalty
        self._niter = args.em_iterations

        # Data-related stuff
        self._load_data(files)
        self._validate_data()
        self._recode_nonseg()
        self._perform_thinning(args.thinning)
        self._normalize_data(args.length_cutoff)

        # Initialize members
        self._init_parameters(args.theta)
        self._init_bounds(args.Nmin)
        self._init_model(args.pieces, args.N0, args.t1,
                args.tK, args.offset, args.knots, args.spline, args.fixed_split)
        self._init_hidden_states(args.initial_model, args.M)
        # TODO re-enable folded mode
        # self._init_inference_manager(args.folded)
        self._init_inference_manager(False)
        self._model.reset()
        self._model.randomize()
        self._init_optimizer(args, files, args.outdir, args.block_size,
                args.fix_rho, args.fixed_split, args.algorithm,
                args.tolerance)

    ## PRIVATE INIT FUNCTIONS
    def _load_data(self, files):
        ## Parse each data set into an array of observations
        logger.info("Loading data...")
        self._files = files
        self._contigs = estimation_tools.parse_text_datasets(files)
        self._npop = max(c.npop for c in self._contigs)
        for c in self._contigs:
            assert len(c.n) == len(c.a)
            assert c.a.max() <= 2
            assert c.a.min() >= 0
            assert c.a.sum() == 2
            assert c.data.shape[1] == 1 + 3 * len(c.n)
        logger.info("%d population%s", self._npop, "" if self._npop == 1 else "s")

    def _init_parameters(self, theta=None, rho=None):
        ## Set theta and rho to their default parameters
        self._L = sum([d[:,0].sum() for d in self._data])
        logger.info("%.2f Gb of data", self._L * 1e-9)
        if theta is not None:
            self._theta = theta
        else:
            # Compute watterson's estimator while not really accounting
            # for any sort of population structure or missing data.
            # TODO This could be improved.
            watt = []
            for c in self._contigs:
                ma = (c.a[None, :] * (c.data[:, 1::3] != -1)).sum(axis=1)
                sample_size = ma + c.data[:, 3::3].sum(axis=1)
                nseg = np.maximum(0, c.data[:, 1::3]).sum(axis=1) + c.data[:, 2::3].sum(axis=1) 
                ss0 = sample_size > 0
                span = c.data[ss0, 0]
                nseg0 = nseg[ss0] > 0
                watt.append((np.average(nseg0 / np.log(sample_size[ss0]), weights=span), len(ss0)))
                assert(np.isfinite(watt[-1][0]))
            x, w = np.array(watt).T
            self._theta = np.average(x, weights=w)
        logger.info("theta: %f", self._theta)
        self._rho = rho or self._theta
        assert np.all(np.isfinite([self._rho, self._theta]))
        logger.info("rho: %f", self._rho)

    def _init_bounds(self, Nmin):
        ## Construct bounds
        # P(seg) is at most theta * 2 * N_max / H_n << 1
        # For now, just base bounds off of population 1. 
        sample_size = 2 + max(sum(c.n) for c in self._contigs)
        Hn = np.log(sample_size)
        Nmax = .1 / (2 * self._theta * Hn)
        logger.debug("Nmax calculated to be %g" % Nmax)
        self._bounds = (Nmin, Nmax)

        # if not args.no_pretrain:
        #     logger.info("Pretraining")
        #     # pretrain if requested
        #     self._pretrain_penalizer = functools.partial(estimation_tools.regularizer, 
        #             penalty=args.pretrain_penalty, f=args.regularizer)
        #     # self._pretrain_penalizer = self._penalizer
        #     self._pretrain(theta, args.folded, args.pretrain_penalty)

    # Optionally thin each dataset
    def _perform_thinning(self, thinning):
        ns = np.array([sum(c.n) for c in self._contigs])
        if isinstance(thinning, int):
            thinning = np.array([thinning] * len(self._contigs))
        if thinning is None:
            thinning = 500 * ns
        if np.any(thinning > 1):
            logger.info("Thinning...")
            self._update_contigs(
                    estimation_tools.thin_dataset(
                        self._data, thinning)
                    )
        elif np.any(ns > 0):
            logger.warn("Not thinning yet undistinguished lineages are present")

    def _init_model(self, pieces, N0, t1, tK, offset,
                    knots, spline_class, fixed_split):
        ## Initialize model
        # FIXME currently disabled.
        # exponential_pieces = args.exponential_pieces or []
        pieces = estimation_tools.extract_pieces(pieces)
        fac = 2. * N0
        t1 /= fac
        tK /= fac
        time_points = estimation_tools.construct_time_points(t1, tK, pieces, offset)
        logger.debug("time points in coalescent scaling:\n%s", str(time_points))
        try:
            num_knots = int(knots)
            knot_spans = np.ones(num_knots, dtype=int)
            knots = np.cumsum(estimation_tools.construct_time_points(t1, tK, knot_spans, offset))
        except ValueError:
            knots = [float(x) for x in knots.split(",")]
        logger.debug("knots in coalescent scaling:\n%s", str(knots))
        spline_class = {"cubic": spline.CubicSpline,
                        "bspline" : spline.BSpline,
                        "akima": spline.AkimaSpline, 
                        "pchip": spline.PChipSpline}[spline_class]
        if self._npop == 1:
            if fixed_split is not None:
                logger.warn("--split was specified, but only one population found in data")
            self._model = SMCModel(time_points, knots, spline_class)
            logger.debug("initial model:\n%s" % np.array_str(self._model[:].astype('float'), precision=3))
        else:
            if fixed_split is not None:
                split = fixed_split
            else:
                split = tK - t1  # just pick the midpoint as a starting value.
            split /= 2. * N0
            self._model = SMCTwoPopulationModel(
                SMCModel(time_points, knots, spline_class),
                SMCModel(time_points, knots, spline_class),
                split)

    def _init_hidden_states(self, initial_model, M):
        cls_d = {cls.__name__: cls for cls in (SMCModel, SMCTwoPopulationModel)}
        if initial_model is not None:
            d = json.load(open(initial_model, "rt"))
            model = cls_d[d['model']['class']].from_dict(d['model'])
        else:
            model = self._model
        ## choose hidden states based on prior model
        dm = model.distinguished_model
        hs = estimation_tools.balance_hidden_states(dm, M)
        self._hidden_states = np.sort(
                np.unique(np.concatenate([self._model.distinguished_model._knots, hs]))
            )
        logger.debug("%d hidden states:\n%s" % (len(self._hidden_states), str(self._hidden_states)))

    def _normalize_data(self, length_cutoff):
        ## break up long spans
        self._contigs, attrs = estimation_tools.break_long_spans(self._contigs, length_cutoff)
        w, het = np.array([a[2:] for k in attrs for a in attrs[k]]).T
        avg = np.average(het, weights=w)
        n = len(het)
        if n == 1:
            avg = 0.
            sd = np.inf
        else:
            var = np.average((het - avg) ** 2, weights=w) * (n / (n - 1.))
            sd = np.sqrt(var)
            logger.debug("Average/sd het:%f(%f)", avg, sd)
            logger.debug("Keeping contigs within +-2 s.d. of mean")
        logger.debug("Average heterozygosity (derived / total bases) by data set (* = dropped)")
        ci = 0
        tpl = "%15d%15d%15d%12g"
        new_contigs = []
        for fn, key in zip(self._files, attrs):
            logger.debug(fn + ":")
            for attr in attrs[key]:
                het = attr[-1]
                mytpl = tpl
                if True or abs(het - avg) <= 3 * sd:
                    new_contigs.append(self._contigs[ci])
                else:
                    mytpl += " *"
                logger.debug(mytpl % attr)
                ci += 1
        self._contigs = new_contigs

    def _validate_data(self):
        for c in self._contigs:
            assert c.data.flags.c_contiguous
            bad = (np.all(c.data[:, 1::3] == c.a[None, :], axis=1) &
                   np.all(c.data[:, 2::3] == c.n[None, :], axis=1))
            if np.any(bad):
                logger.error("In file %s, observations %s:", c.fn, np.where(bad)[0])
                raise RuntimeError("Error: data set contains sites where every "
                        "individual is homozygous recessive. Please encode / "
                        "fold these as non-segregating (homozygous dominant).")

    def _recode_nonseg(self):
        self._contigs = estimation_tools.recode_nonseg(self._contigs)

    def _init_inference_manager(self, folded):
        ## Create inference object which will be used for all further calculations.
        logger.debug("Creating inference manager...")
        self._ims = {}
        if self._npop == 1:
            n = max(c.n[0] for c in self._contigs)
            k = (n, None)
            im = _smcpp.PyOnePopInferenceManager(n, self._data, self._hidden_states)
            self._ims[k] = im
            im.model = self.model
            im.theta = self._theta
            im.rho = self._rho
        elif self._npop == 2:
            contig_d = {}
            pop1 = True
            for c in self._contigs:
                if c.npop == 1:
                    assert c.a[0] == 2
                    if pop1:
                        k = ((2, c.n[0]), None)
                    else:
                        k = (None, (2, c.n[0]))
                else:
                    pop1 = False
                    k = (tuple(c.n), tuple(c.a))
                contig_d.setdefault(k, []).append(c)
            for k in contig_d:
                data = [c.data for c in contig_d[k]]
                if None in k:  # one population case
                    n = k[0][1] if k[1] is None else k[1][1]
                    im = _smcpp.PyOnePopInferenceManager(n, data, self._hidden_states)
                    im.model = self.model.model1 if k[1] is None else self.model.model2
                else:
                    im = _smcpp.PyTwoPopInferenceManager(k[0][0], k[0][1],
                            k[1][0], k[1][1], data, self._hidden_states)
                    im.model = self.model
                im.theta = self._theta
                im.rho = self._rho
                self._ims[k] = im
        else:
            logger.error("Only 1 or 2 populations are supported at this time")
            sys.exit(1)

    def _init_optimizer(self, args, files, outdir, block_size,
            fix_rho, fixed_split, algorithm, tolerance):
        if self._npop == 1:
            self._optimizer = optimizer.SMCPPOptimizer(self, algorithm, tolerance)
            # Also optimize knots in 1 pop case. Not yet implemented
            # for two pop case.
            # self._optimizer.register(optimizer.KnotOptimizer())
        elif self._npop == 2:
            self._optimizer = optimizer.TwoPopulationOptimizer(self, algorithm, tolerance)
            if fixed_split is None:
                smax = np.sum(self._model.distinguished_model.s)
                self._optimizer.register(optimizer.ParameterOptimizer("split", (0., smax), "model"))
        self._optimizer.block_size = block_size
        self._optimizer.register(optimizer.AnalysisSaver(outdir))
        if not fix_rho:
            self._optimizer.register(optimizer.ParameterOptimizer("rho", (1e-6, 1e-2)))

    def _update_contigs(self, new_data):
        assert len(self._contigs) == len(new_data)
        self._contigs = [Contig(data=d, n=c.n, a=c.a)
                for c, d in zip(self._contigs, new_data)]

    @property
    def _data(self):
        return [c.data for c in self._contigs]
    ## END OF PRIVATE FUNCTIONS

    ## PUBLIC INTERFACE
    @logging.log_step("Running optimizer...", "Optimization completed.")
    def run(self):
        'Perform the analysis.'
        self._optimizer.run(self._niter)

    def Q(self, k=None):
        'Value of Q() function in M-step.'
        # q1, q2, q3 = self._im.Q(True)
        qq = sum([im.Q() for im in self._ims.values()])
        qr = -self._penalty * self.model.regularizer()
        logger.debug(("im.Q", float(qq), [qq.d(x) for x in self.model.dlist]))
        logger.debug(("reg", float(qr), [qr.d(x) for x in self.model.dlist]))
        return qq + qr

    @logging.log_step("Running E-step...", "E-step completed.")
    def E_step(self):
        'Perform E-step.'
        for im in self._ims.values():
            im.E_step()

    def loglik(self):
        'Log-likelihood of data after most recent E-step.'
        ll = sum(im.loglik() for im in self._ims.values())
        return ll - self._penalty * float(self.model.regularizer())

    @property
    def model(self):
        return self._model

    @property
    def bounds(self):
        return self._bounds

    @property
    def npop(self):
        'The number of populations contained in this analysis.'
        return self._npop

    @property
    def N0(self):
        return self._N0

    @property
    def rho(self):
        return self._rho

    @rho.setter
    def rho(self, r):
        self._rho = r
        for im in self._ims.values():
            im.rho = r

    def dump(self, filename):
        'Dump result of this analysis to :filename:.'
        d = {'N0': self._N0, 'theta': self._theta, 'rho': self._rho}
        d['model'] = self.model.to_dict()
        json.dump(d, open(filename + ".json", "wt"), sort_keys=True, indent=4)
