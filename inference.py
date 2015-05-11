import numpy as np
import multiprocessing

import _pypsmcpp


def parallel_sfs(*args):
    return _pypsmcpp.sfs(demo, S, M, n, tau1, tau2, False) 

def log_p(demo, S, M, n, obs, theta, rho, hidden_states):
    '''Return probability of observing <obs> under demography <demo>, as
    computed by forward algorithm.'''
    sfss = []
    for tau1, tau2 in zip(hidden_states[:-1], hidden_states[1:]):
        sfss.append([_pypsmcpp.sfs(demo, S, M, n, tau1, tau2, False)])
    _validate_obs(n, obs)
    return _pypsmcpp.hmm(demo, sfss, obs, hidden_states, rho, theta)

def _validate_obs(n, obs):
    os = obs.sum(axis=1)
    mx = np.max(obs, axis=0)
    if any([not np.all([0 < os, os < n]), mx[0] < 0, mx[0] > 2, mx[1] < 0, mx[1] > n - 1]):
        raise RuntimeError("invalid?")    
