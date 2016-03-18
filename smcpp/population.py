from __future__ import absolute_import, division, print_function
import numpy as np
import jsonpickle
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
        t1 = args.t1 / (2 * args.N0)
        tK = args.tK / (2 * args.N0)
        pieces = estimation_tools.extract_pieces(args.pieces)
        time_points = estimation_tools.construct_time_points(t1, tK, pieces)
        logger.debug("time points in coalescent scaling:\n%s", str(time_points))
        K = len(time_points)

        ## Construct bounds
        Nmin = args.Nmin / (2 * args.N0)
        Nmax = args.Nmax / (2 * args.N0)
        self._bounds = np.array([[Nmin, Nmax]] * K + 
                [[1.01 * Nmin, 0.99 * Nmax]] * K).reshape([2, K, 2])

        ## Parse each data set into an array of observations
        logger.info("Loading data...")
        dataset = util.parse_text_datasets(dataset_files)
        n = 2 + max([obs[:, -1].max() for obs in dataset])
        ## At this point, data have not been thinned or anything. 

        ## Initialize model
        exponential_pieces = args.exponential_pieces or []
        self._model = SMCModel(time_points, exponential_pieces)

        ## Set theta and rho to their default parameters
        if args.theta is not None:
            self._theta = args.theta
        else:
            L = sum([obs[:,0].sum() for obs in dataset])
            Lseg = 0
            for obs in dataset:
                conds = (obs[:, 1:3].sum(axis=1) > 0) & (obs[:, 3] == n - 2) & (obs[:, 1] > -1)
                Lseg += conds.sum()
            segfrac = 1. * Lseg / L
            self._theta = segfrac / (1. / np.arange(1, n)).sum()
            logger.info("watterson's theta: %f" % self._theta)
        self._rho = args.rho or self._theta / 4.

        ## After (potentially) doing pretraining, normalize and thin the data set
        ## Optionally thin each dataset
        if args.thinning is not None:
            logger.info("Thinning...")
            dataset = estimation_tools.thin_dataset(dataset, args.thinning)
        
        # Prepare empirical SFS for later use. This is cheap to compute
        esfs = util.compute_esfs(dataset, n)
        self._sfs = np.mean(esfs, axis=0)

        # pretrain if requested
        self._penalizer = functools.partial(estimation_tools.regularizer, 
                penalty=args.regularization_penalty, f=args.regularizer)

        if not args.no_pretrain:
            logger.info("Pretraining")
            self._pretrain(self._theta)
    
        # We remember the initialized model for use in split estimated
        self._init_model_x = self._model.x.copy()

        ## choose hidden states based on prior model
        logger.info("Balancing hidden states...")
        self._balance_hidden_states(args.M)

        ## break up long spans
        self._dataset, attrs = estimation_tools.break_long_spans(dataset, 
                args.span_cutoff, args.length_cutoff)

        logger.debug("Average heterozygosity (derived / total bases) by data set:")
        for fn, key in zip(dataset_files, attrs):
            logger.debug(fn + ":")
            for attr in attrs[key]:
                logger.debug("%15d%15d%15d%12g%12g" % attr)

        ## Create inference object which will be used for all further calculations.
        logger.debug("Creating inference manager...")
        self._im = _smcpp.PyInferenceManager(n - 2, self._dataset, self._hidden_states)
        self._im.model = self._model
        self._im.theta = self._theta
        self._im.rho = self._rho

    def _balance_hidden_states(self, M):
        hs = _smcpp.balance_hidden_states(self._model, M)
        cs = np.cumsum(self._model.s)
        cs = cs[cs <= hs[1]]
        self._hidden_states = np.sort(np.unique(np.concatenate([cs, hs])))
        logger.info("hidden states:\n%s" % str(self._hidden_states))

    def reset(self):
        self._model.x[:] = self._init_model_x[:]

    def penalize(self, model):
        return self._penalizer(model)

    def _pretrain(self, theta):
        estimation_tools.pretrain(self._model, self._sfs, self._bounds, theta, self.penalize)

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

    model = _tied_property("model")
    theta = _tied_property("theta")
    rho = _tied_property("rho")
    derivatives = _tied_property("derivatives")

    def dump(self, fn):
        er = EstimationResult()
        for attr in ['model', 'N0']:
            setattr(er, attr, getattr(self, "_" + attr))
        er.dump(fn)
