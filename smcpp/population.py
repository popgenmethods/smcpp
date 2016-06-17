from __future__ import absolute_import, division, print_function
import numpy as np
import functools
import multiprocessing
from logging import getLogger
logger = getLogger(__name__)

from . import estimation_tools, _smcpp, util
from .estimation_result import EstimationResult
from .model import SMCModel


def _tied_property(attr):
    def getx(self):
        return getattr(self._im, attr)
    def setx(self, x):
        setattr(self._im, attr, x)
    return property(getx, setx)
        

class Population(object):
    '''Class representing a population + model for estimation.'''
    def __init__(self, dataset_files, args):
        self._N0 = args.N0
        t1 = np.array(args.t1) / (2 * args.N0)
        tK = args.tK / (2 * args.N0)
        pieces = estimation_tools.extract_pieces(args.pieces)
        time_points = estimation_tools.construct_time_points(t1, tK, pieces)
        logger.debug("time points in coalescent scaling:\n%s", str(time_points))
        K = len(time_points)

        ## Parse each data set into an array of observations
        logger.info("Loading data...")
        dataset = util.parse_text_datasets(dataset_files)
        n = 2 + max([obs[:, -1].max() for obs in dataset])
        logger.info("n=%d" % n)
        ## At this point, data have not been thinned or anything. 

        ## Initialize model
        exponential_pieces = args.exponential_pieces or []
        self._model = SMCModel(time_points, np.cumsum(time_points)[::4])

        ## Set theta and rho to their default parameters
        self._L = sum([obs[:,0].sum() for obs in dataset])
        logger.info("%.2f Gb of data" % (self._L * 1e-9))
        if args.theta is not None:
            theta = args.theta
        else:
            Lseg = 0
            for obs in dataset:
                conds = (obs[:, 1:3].sum(axis=1) > 0) & (obs[:, 3] == n - 2) & (obs[:, 1] > -1)
                Lseg += conds.sum()
            segfrac = 1. * Lseg / self._L
            theta = segfrac / (1. / np.arange(1, n)).sum()
        logger.info("theta: %f" % theta)
        rho = args.rho or theta / 4.
        logger.info("rho: %f" % rho)

        ## Construct bounds
        # P(seg) is at most theta * 2 * N_max / H_n << 1
        Hn = (1. / np.arange(1, n)).sum()
        Nmax = .1 / (2 * theta * Hn)
        logger.debug("Nmax calculated to be %g" % Nmax)
        self._bounds = np.array([[args.Nmin, Nmax]] * K + 
                [[1.01 * args.Nmin, 0.99 * Nmax]] * K).reshape([2, K, 2])
        # logger.debug("bounds:\n%s" % np.array_str(self._bounds, precision=3))

        ## After (potentially) doing pretraining, normalize and thin the data set
        ## Optionally thin each dataset
        if args.thinning is None:
            args.thinning = 400 * n
        if args.thinning > 1:
            logger.info("Thinning...")
            dataset = estimation_tools.thin_dataset(dataset, args.thinning)
        elif n > 2:
            logger.warn("Not thinning yet n=%d. This probably isn't what you desire, see --thinning" % (n//2,))
        
        # Prepare empirical SFS for later use. This is cheap to compute
        esfs = util.compute_esfs(dataset, n)
        self._sfs = np.sum(esfs, axis=0) / np.sum(esfs)
        logger.info("Empirical SFS:\n%s" % np.array_str(self._sfs, precision=4))
        logger.info("Reduced SFS:\n%s" % np.array_str(util.undistinguished_sfs(self._sfs, args.folded), precision=4))

        if args.regularization_penalty is None:
            args.regularization_penalty = 3e-9 * self._L
        logger.info("regularizer: %s" % args.regularizer)
        logger.info("regularization penalty: %g" % args.regularization_penalty)

        self._penalizer = functools.partial(estimation_tools.regularizer, 
                penalty=args.regularization_penalty, f=args.regularizer)

        if not args.no_pretrain:
            logger.info("Pretraining")
            # pretrain if requested
            self._pretrain_penalizer = functools.partial(estimation_tools.regularizer, 
                    penalty=args.pretrain_penalty, f=args.regularizer)
            # self._pretrain_penalizer = self._penalizer
            self._pretrain(theta, args.folded)
    
        # We remember the initialized model for use in split estimation
        if args.init_model is not None:
            er = EstimationResult.load(args.init_model)
            self._model.y = er.model.y
            rho = er.rho
            theta = er.theta
        self._init_model_y = self._model.y.copy() 
        logger.debug("initial model:\n%s" % np.array_str(self._model.y, precision=3))

        ## choose hidden states based on prior model
        if args.hidden_states:
            self._hidden_states = args.hidden_states
        else:
            logger.info("Balancing hidden states...")
            self._model._spline.eval(0.0)
            self._balance_hidden_states(args.M)

        ## break up long spans
        self._dataset, attrs = estimation_tools.break_long_spans(dataset, rho, args.length_cutoff)

        logger.debug("Average heterozygosity (derived / total bases) by data set:")
        for fn, key in zip(dataset_files, attrs):
            logger.debug(fn + ":")
            for attr in attrs[key]:
                logger.debug("%15d%15d%15d%12g%12g" % attr)

        ## Create inference object which will be used for all further calculations.
        logger.debug("Creating inference manager...")
        self._im = _smcpp.PyInferenceManager(n - 2, self._dataset, self._hidden_states, time_points)
        self._im.model = self._model
        self._im.theta = theta
        self._im.rho = rho
        self._im.folded = args.folded

    def _balance_hidden_states(self, M):
        hs = _smcpp.balance_hidden_states(self._model, M)
        cs = np.cumsum(self._model.s)
        cs = cs[cs <= hs[1]]
        self._hidden_states = np.sort(np.unique(np.concatenate([cs, hs])))
        logger.info("%d hidden states:\n%s" % (len(self._hidden_states), str(self._hidden_states)))

    def reset(self):
        self.model.x[:] = self._init_model_x[:]

    def _pretrain(self, theta, folded):
        estimation_tools.pretrain(self._model, self._sfs, self._bounds, theta, folded, args.pretrain_penalty)

    def randomize(self):
        for i in range(2):
            for j in range(self.model.x.shape[1]):
                self.model.x[i, j] = np.random.uniform(*self.bounds[i, j])

    def sfs(self):
        return self._sfs

    def Q(self):
        return self._im.Q()

    def E_step(self):
        return self._im.E_step()

    def loglik(self):
        return self._im.loglik()

    def precond(self):
        return self.model.precond

    @property
    def bounds(self):
        return self._bounds

    @property
    def N0(self):
        return self._N0

    model = _tied_property("model")
    theta = _tied_property("theta")
    rho = _tied_property("rho")
    derivatives = _tied_property("derivatives")

    def dump(self, fn):
        er = EstimationResult()
        for attr in ['model', 'N0', 'theta', 'rho']:
            setattr(er, attr, getattr(self, attr))
        er.dump(fn)
