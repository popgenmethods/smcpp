import numpy as np
import functools
import multiprocessing
import inflect
import json
import sys
import os.path

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
        self._load_data(files)
        self._init_parameters(args.theta, args.rho)
        self._init_model(args.initial_model, args.pieces, args.N0, args.t1, args.tK, args.knots, args.spline, args.split)
        self._init_hidden_states(args.M)
        self._model.reset()
        self._init_bounds(args.Nmin)
        self._perform_thinning(args.thinning)
        self._normalize_data(args.length_cutoff)
        self._init_inference_manager(args.folded)
        self._init_optimizer(args.outdir, args.block_size, args.fix_rho, args.fix_split)

        # Misc. parameter initialiations
        self._N0 = args.N0
        self._penalty = args.regularization_penalty
        self._niter = args.em_iterations
    
    ## PRIVATE INIT FUNCTIONS
    def _load_data(self, files):
        ## Parse each data set into an array of observations
        logger.info("Loading data...")
        self._files = files
        self._data = data = util.parse_text_datasets(files)
        self._npop = data[0].shape[1] // 2 - 1
        for d in data:
            assert d.shape[1] == 2 * (self._npop + 1)
        logger.info(inflect.engine().no("population", self._npop))
        self._n = np.max([np.max(obs[:, 2::2], axis=0) for obs in self._data], axis=0)
        logger.info("n=%s" % self._n)

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
        Hn = np.log(self._n.sum())
        Nmax = .1 / (2 * self._theta * Hn)
        logger.debug("Nmax calculated to be %g" % Nmax)
        self._bounds = (Nmin, Nmax)

        # Prepare empirical SFS for later use. This is cheap to compute
        # esfs = util.compute_esfs(dataset, n)
        # self._sfs = np.sum(esfs, axis=0) / np.sum(esfs)
        # logger.info("Empirical SFS:\n%s" % np.array_str(self._sfs, precision=4))
        # logger.info("Reduced SFS:\n%s" % np.array_str(util.undistinguished_sfs(self._sfs, args.folded), precision=4))

        # if not args.no_pretrain:
        #     logger.info("Pretraining")
        #     # pretrain if requested
        #     self._pretrain_penalizer = functools.partial(estimation_tools.regularizer, 
        #             penalty=args.pretrain_penalty, f=args.regularizer)
        #     # self._pretrain_penalizer = self._penalizer
        #     self._pretrain(theta, args.folded, args.pretrain_penalty)
    
        ## After (potentially) doing pretraining, normalize and thin the data set
        ## Optionally thin each dataset
    def _perform_thinning(self, thinning):
        if thinning is None:
            thinning = 400 * np.sum(self._n)
        if thinning > 1:
            logger.info("Thinning...")
            self._data = estimation_tools.thin_dataset(self._data, thinning)
        elif self._n.sum() > 2:
            logger.warn("Not thinning yet n = %d > 0. This probably "
                        "isn't what you desire, see --thinning", self._n.sum() // 2 + 1)
        
    def _init_model(self, initial_model, pieces, N0, t1, tK, num_knots, spline_class, split=None):
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
        t1 = np.array(t1)
        fac = 2. * N0
        t1 /= fac
        tK /= fac
        time_points = estimation_tools.construct_time_points(t1, tK, pieces)
        logger.debug("time points in coalescent scaling:\n%s", str(time_points))
        knots = np.cumsum(estimation_tools.construct_time_points(t1, tK, [1] * num_knots))
        logger.debug("knots in coalescent scaling:\n%s", str(knots))
        spline_class = {"cubic": spline.CubicSpline,
                        "akima": spline.AkimaSpline, 
                        "pchip": spline.PChipSpline}[spline_class]
        if self._npop == 1:
            if split is not None:
                logger.warn("--split was specified, but only one population found in data")
            self._model = SMCModel(time_points, knots, spline_class)
            logger.debug("initial model:\n%s" % np.array_str(self._model.y, precision=3))
        else:
            if split is None:
                raise RuntimeError("Initial value of split must be specified for two-population model")
            split /= 2. * N0
            self._model = SMCTwoPopulationModel(
                SMCModel(time_points, knots, spline_class),
                SMCModel(time_points, knots, spline_class),
                split)

    def _init_hidden_states(self, M):
        ## choose hidden states based on prior model
        hs = estimation_tools.balance_hidden_states(self._model.distinguished_model, M)
        cs = np.cumsum(self._model.distinguished_model.s)
        cs = cs[cs <= hs[1]]
        self._hidden_states = np.sort(np.unique(np.concatenate([cs, hs])))
        logger.debug("%d hidden states:\n%s" % (len(self._hidden_states), str(self._hidden_states)))

    def _normalize_data(self, length_cutoff):
        ## break up long spans
        self._data, attrs = estimation_tools.break_long_spans(self._data, self._rho, length_cutoff)
        logger.debug("Average heterozygosity (derived / total bases) by data set:")
        for fn, key in zip(self._files, attrs):
            logger.debug(fn + ":")
            for attr in attrs[key]:
                logger.debug("%15d%15d%15d%12g%12g" % attr)

    def _init_inference_manager(self, folded):
        ## Create inference object which will be used for all further calculations.
        logger.debug("Creating inference manager...")
        if self._npop == 1:
            self._im = _smcpp.PyOnePopInferenceManager(self._n[0], self._data, self._hidden_states)
        elif self._npop == 2:
            self._im = _smcpp.PyTwoPopInferenceManager(self._n[0], self._n[1], self._data, self._hidden_states)
            # self._jcsfs = jcsfs.JointCSFS(self._n[0], self._n[1], 2, 0, self._hidden_states)
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

    def _init_optimizer(self, outdir, block_size, fix_rho, fix_split):
        if self._npop == 1:
            self._optimizer = optimizer.SMCPPOptimizer(self)
        elif self._npop == 2:
            self._optimizer = optimizer.TwoPopulationOptimizer(self)
            if not fix_split:
                smax = np.sum(self.model.distinguished_model.s)
                self._optimizer.register(optimizer.ParameterOptimizer("split", (0., smax), "model"))
        self._optimizer.block_size = block_size
        self._optimizer.register(optimizer.AnalysisSaver(outdir))
        if not fix_rho:
            self._optimizer.register(optimizer.ParameterOptimizer("rho", (1e-6, 1e-2)))

    # FIXME re-enable this
    # def _pretrain(self, theta, folded, penalty):
    #     estimation_tools.pretrain(self._model, self._sfs, self._bounds, theta, folded, penalty)
    ## END OF PRIVATE FUNCTIONS

    ## PUBLIC INTERFACE
    @logging.log_step("Running optimizer...", "Optimization completed.")
    def run(self):
        'Perform the analysis.'
        self._optimizer.run(self._niter)

    def Q(self):
        'Value of Q() function in M-step.'
        return self._im.Q() - self._penalty * self.model.regularizer()

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
