import numpy as np
import logging
import moran_model

logger = logging.getLogger(__name__)

import _pypsmcpp

def logp(sqrt_a, b, sqrt_s, S, M, n, obs_list, theta, rho, hidden_states, numthreads=1):
    '''Return probability of observing <obs> under demography <demo>, as
    computed by forward algorithm.'''
    demo = _pypsmcpp.Demography(sqrt_a, b, sqrt_s)
    sfss = [_pypsmcpp.sfs(demo, S, M, n, tau1, tau2, theta, numthreads=numthreads)
            for tau1, tau2 in zip(hidden_states[:-1], hidden_states[1:])]
    for obs in obs_list:
        _validate_obs(n, obs)
    return _pypsmcpp.hmm(demo, sfss, obs_list, hidden_states, rho, theta, numthreads)

def _validate_obs(n, obs):
    sfs = obs[:, 1:]
    os = sfs.sum(axis=1)
    mx = np.max(sfs, axis=0)
    if any([not np.all([0 <= os, os < n]), mx[0] < 0, mx[0] > 2, mx[1] < 0, mx[1] > n - 1]):
        raise RuntimeError("invalid?")    
