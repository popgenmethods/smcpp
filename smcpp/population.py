import numpy as np
import logging
logger = logging.getLogger(__name__)

from .estimation_tools import compute_empirical_sfs

class Population(object):
    '''Class representing a population + model for estimation.'''
    def __init__(self, dataset, time_points, exponential_pieces, theta, rho, M, bounds, pretrain):
        self._dataset = dataset
        self._n = 2 + max([obs[:, -1].max() for obs in dataset])
        self._time_point = time_points
        self._exponential_pieces = exponential_pieces
        self._theta = theta
        self._rho = rho
        self._bounds = bounds

        # Prepare empirical SFS for later use. This is cheap to compute
        self._obsfs = sum([empirical_sfs(ol, n) for ol in obs_list])
        self._obsfs /= self._obsfs.sum()

        ## Initialize model
        self._model = SMCModel(time_points, exponential_pieces)

        # pretrain if requested
        if pretrain:
            self._pretrain()

        self._balance_hidden_states()

        ## Create inference object which will be used for all further calculations.
        self._im = _smcpp.PyInferenceManager(self._n - 2, self._obs_list, 
                self._hidden_states, self._theta, self._rho)

    @property
    def M(self):
        return len(self._hidden_states) - 1

    def _balance_hidden_states(self):
        hs = _smcpp.balance_hidden_states(self._model.x, self.M)
        cs = np.cumsum(self._model.s)
        cs = cs[cs <= hs[1]]
        self._hidden_states = np.sort(np.unique(np.concatenate([cs, hs])))

    def _pretrain(self, penalty):
        self._model.x[:] = estimation_tools.pretrain(self._model, self._obsfs, self._bounds, penalty)

    def E_step(self, fbonly):
        return self._im.E_step(fbonly)

    def set_params(self, params, deriv):
        return self._im.set_params(params, deriv)

    def loglik(self):
        return self._im.loglik()
