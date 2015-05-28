import numpy as np
import logging
import moran_model

logger = logging.getLogger(__name__)

import _pypsmcpp

def loglik(xdiff, sqrt_y, n, S, M, obs_list, hidden_states, rho, theta, 
        reg, numthreads=1, seed=None, viterbi=False, jacobian=False):
    '''Return probability of observing <obs> under demography <demo>, as
    computed by forward algorithm.'''
    for obs in obs_list:
        _validate_obs(n, obs)
    return _pypsmcpp.log_likelihood(xdiff, sqrt_y, n, S, M, obs_list, hidden_states, rho, theta, 
            reg, numthreads, seed, viterbi, jacobian)

def _validate_obs(n, obs):
    sfs = obs[:, 1:]
    os = sfs.sum(axis=1)
    mx = np.max(sfs, axis=0)
    if any([not np.all([0 <= os, os < n + 2]), mx[0] < 0, mx[0] > 2, mx[1] < 0, mx[1] > n]):
        raise RuntimeError("invalid?")
