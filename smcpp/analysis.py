import numpy as np
import functools
import json
import sys
import os.path
import scipy.optimize
import ad

from . import estimation_tools, _smcpp, util, logging, optimizer, jcsfs, spline
from .model import SMCModel, SMCTwoPopulationModel
from .observe import Observer

logger = logging.getLogger(__name__)

def _tied_property(attr):
    def getx(self):
        return getattr(self._im, attr)
    def setx(self, x):
        setattr(self._im, attr, x)
    return property(getx, setx)
        

class Analysis(Observer):
    '''A dataset, model and inference manager to be used for estimation.'''
    def __init__(self, files, args):
        # Misc. parameter initialiations
        self._N0 = args.N0
        self._penalty = args.regularization_penalty
        self._niter = args.em_iterations
        self._load_data(files)
        self._init_parameters(args.theta, args.rho)
        self._init_bounds(args.Nmin)
        self._init_model(args.initial_model, args.pieces, args.N0, args.t1,
                args.tK, args.offset, args.knots, args.spline, args.fixed_split)
        self._model.reset()
        # Add a small amount of noise to model. If all pieces are equal,
        # the derivatives can be off due to some branching in the spline
        # code.
        self._model.randomize()
        self._init_optimizer(args, files, args.outdir, args.block_size,
                args.fix_rho, args.fixed_split, args.algorithm,
                args.tolerance)
        self._init_hidden_states(args.M)
        self._perform_thinning(args.thinning)
        self._normalize_data(args.length_cutoff)
        self._init_inference_manager(args.folded)

    ## PRIVATE INIT FUNCTIONS
    def _load_data(self, files):
        ## Parse each data set into an array of observations
        logger.info("Loading data...")
        self._files = files
        self._data = util.parse_text_datasets(files)
        self._npop = (self._data[0].shape[1] - 1) / 3
        for d in self._data:
            assert d.shape[1] == 1 + 3 * self._npop
        logger.info("%d population%s", self._npop, "" if self._npop == 1 else "s")
        self._n = np.max([np.max(obs[:, 3::3], axis=0) for obs in self._data], axis=0)
        if self._npop == 1:
            self._a = np.array([2])
        else:
            self._a = np.max([np.max(obs[:, 1::3], axis=0) for obs in self._data], axis=0)
        logger.info("n=%s" % self._n)
        logger.info("a=%s" % self._a)

    def _init_parameters(self, theta=None, rho=None):
        ## Set theta and rho to their default parameters
        self._L = sum([obs[:,0].sum() for obs in self._data])
        logger.info("%.2f Gb of data", self._L * 1e-9)
        if theta is not None:
            self._theta = theta
        else:
            # Compute watterson's estimator while not really accounting
            # for any sort of population structure or missing data.
            # TODO This could probably be improved.
            Lseg = 0
            for obs in self._data:
                conds = (
                        (obs[:, 0] == 1) &
                        ((obs[:, 1] > 0) | (obs[:, 2::2].max(axis=1) > 0))
                        )
                Lseg += conds.sum()
            segfrac = 1. * Lseg / self._L
            self._theta = segfrac / (1. / np.arange(1, np.sum(self._n))).sum()
        logger.info("theta: %f", self._theta)
        self._rho = rho or self._theta
        logger.info("rho: %f", self._rho)

    def _init_bounds(self, Nmin):
        ## Construct bounds
        # P(seg) is at most theta * 2 * N_max / H_n << 1
        # For now, just base bounds off of population 1. 
        sample_size = self._n.sum() + self._a.sum()
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
        if thinning is None:
            thinning = 400 * np.sum(self._n)
        if thinning > 1:
            logger.info("Thinning...")
            self._data = estimation_tools.thin_dataset(self._data, thinning)
        elif self._n.sum() > 2:
            logger.warn("Not thinning yet n = %d > 0. This probably "
                        "isn't what you desire, see --thinning", self._n.sum() // 2 + 1)

    def _init_model(self, initial_model, pieces, N0, t1, tK, offset,
                    knots, spline_class, fixed_split):
        if initial_model is not None:
            d = json.load(open(initial_model, "rt"))
            if self._npop == 1:
                klass = SMCModel
            else:
                klass = SMCTwoPopulationModel
            self._model = klass.from_dict(d['model'])
            return
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
            knots = np.cumsum(estimation_tools.construct_time_points(t1, tK, [1] * num_knots, offset))
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

    def _init_hidden_states(self, M):
        ## choose hidden states based on prior model
        dm = self._model.distinguished_model
        hs = estimation_tools.balance_hidden_states(dm, M)
        self._hidden_states = np.sort(
                np.unique(np.concatenate([self._model.distinguished_model._knots, hs]))
            )
        logger.debug("%d hidden states:\n%s" % (len(self._hidden_states), str(self._hidden_states)))

    def _normalize_data(self, length_cutoff):
        ## break up long spans
        self._data, attrs = estimation_tools.break_long_spans(self._data, self._rho, length_cutoff)
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
        dsi = 0
        tpl = "%15d%15d%15d%12g"
        new_data = []
        for fn, key in zip(self._files, attrs):
            logger.debug(fn + ":")
            for attr in attrs[key]:
                het = attr[-1]
                mytpl = tpl
                if abs(het - avg) <= 3 * sd:
                    new_data.append(self._data[dsi])
                else:
                    mytpl += " *"
                logger.debug(mytpl % attr)
                dsi += 1
        self._data = new_data


    def _init_inference_manager(self, folded):
        ## Create inference object which will be used for all further calculations.
        logger.debug("Creating inference manager...")
        if self._npop == 1:
            self._im = _smcpp.PyOnePopInferenceManager(self._n[0], self._data, self._hidden_states)
        elif self._npop == 2:
            self._im = _smcpp.PyTwoPopInferenceManager(self._n[0], self._n[1],
                    self._a[0], self._a[1], self._data, self._hidden_states)
        else:
            logger.error("Only 1 or 2 populations are supported at this time")
            sys.exit(1)
        self._im.model = self._model
        # Model should completely live in the IM now
        del self._model
        self._im.theta = self._theta
        self._im.rho = self._rho
        self._im.folded = folded
        # Receive updates whel model changes
        self.model.register(self)

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

    ## END OF PRIVATE FUNCTIONS

    ## PUBLIC INTERFACE
    @logging.log_step("Running optimizer...", "Optimization completed.")
    def run(self):
        'Perform the analysis.'
        self._optimizer.run(self._niter)

    def Q(self, k=None):
        'Value of Q() function in M-step.'
        # q1, q2, q3 = self._im.Q(True)
        qq = self._im.Q()
        qr = -self._penalty * self.model.regularizer()
        logger.debug(("im.Q", float(qq), [qq.d(x) for x in self.model.dlist]))
        logger.debug(("reg", float(qr), [qr.d(x) for x in self.model.dlist]))
        return qq + qr

    @logging.log_step("Running E-step...", "E-step completed.")
    def E_step(self):
        'Perform E-step.'
        return self._im.E_step()

    def loglik(self):
        'Log-likelihood of data after most recent E-step.'
        return self._im.loglik() - self._penalty * float(self.model.regularizer())

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

    model = _tied_property("model")
    theta = _tied_property("theta")
    rho = _tied_property("rho")

    def update(self, message, *args, **kwargs):
        'Keep inference manager and model in sync by listening for model changes.'
        if message == "model update":
            self._im.model = self.model

    def dump(self, filename):
        'Dump result of this analysis to :filename:.'
        d = {'N0': self._N0, 'theta': self._theta, 'rho': self._rho}
        d['model'] = self.model.to_dict()
        json.dump(d, open(filename + ".json", "wt"), sort_keys=True, indent=4)
