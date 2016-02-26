import numpy as np
import logging
import jsonpickle
logger = logging.getLogger(__name__)

from . import estimation_tools, _smcpp
from .estimation_result import EstimationResult
from .model import SMCModel

class Population(object):
    '''Class representing a population + model for estimation.'''
    def __init__(self, dataset, time_points, exponential_pieces, N0, theta, rho, M, bounds, pretrain):
        self._dataset = dataset
        self._n = 2 + max([obs[:, -1].max() for obs in dataset])
        self._time_points = time_points
        self._exponential_pieces = exponential_pieces
        self._N0 = N0
        self._theta = theta
        self._rho = rho
        self._M = M
        self._bounds = bounds

        # Prepare empirical SFS for later use. This is cheap to compute
        self._obsfs = sum([estimation_tools.empirical_sfs(obs, self._n) for obs in dataset])
        self._obsfs /= self._obsfs.sum()

        ## Initialize model
        self._model = SMCModel(time_points, exponential_pieces)

        # pretrain if requested
        if pretrain:
            self._pretrain(0.)

        self._balance_hidden_states()

        ## Create inference object which will be used for all further calculations.
        self._im = _smcpp.PyInferenceManager(self._n - 2, self._dataset,
                self._hidden_states, self._theta, self._rho)

    def _balance_hidden_states(self):
        hs = _smcpp.balance_hidden_states(self._model, self._M)
        cs = np.cumsum(self._model.s)
        cs = cs[cs <= hs[1]]
        self._hidden_states = np.sort(np.unique(np.concatenate([cs, hs])))
        logging.info("hidden states:\n%s" % str(self._hidden_states))

    def _pretrain(self, penalty):
        estimation_tools.pretrain(self._model, self._obsfs, self._bounds, self._theta, penalty)

    def theta(self):
        return self._theta

    def obsfs(self):
        return self._obsfs

    def Q(self):
        return self._im.Q()

    def E_step(self, fbonly):
        return self._im.E_step(fbonly)

    def set_params(self, model, deriv):
        return self._im.set_params(model, deriv)

    def loglik(self):
        return self._im.loglik()

    def precond(self):
        return self.model().precond

    def model(self):
        return self._model

    def dump(self, fn):
        er = EstimationResult()
        for attr in ['model', 'theta', 'rho', 'N0']:
            setattr(er, attr, getattr(self, "_" + attr))
        er.dump(fn)
