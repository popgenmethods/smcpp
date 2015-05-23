import numpy as np
import logging
import moran_model

logger = logging.getLogger(__name__)

import _pypsmcpp

def loglik(a, b, s, S, M, n, obs_list, theta, rho, hidden_states, numthreads=1, 
        viterbi=False, reg_a=0, reg_b=0, reg_s=0):
    '''Return probability of observing <obs> under demography <demo>, as
    computed by forward algorithm.'''
    for obs in obs_list:
        _validate_obs(n, obs)
    return _pypsmcpp.log_likelihood(a, b, s, n, S, M, obs_list, hidden_states, rho, theta, reg_a, reg_b, reg_s)

def _validate_obs(n, obs):
    sfs = obs[:, 1:]
    os = sfs.sum(axis=1)
    mx = np.max(sfs, axis=0)
    if any([not np.all([0 <= os, os < n]), mx[0] < 0, mx[0] > 2, mx[1] < 0, mx[1] > n - 1]):
        raise RuntimeError("invalid?")
