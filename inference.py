from __future__ import division
import numpy as np
import logging

logger = logging.getLogger(__name__)

from _pypsmcpp import PyInferenceManager
from util import grouper
from scrm import tree_obs_iter

def posterior_decode_score(l1, l2, block_size, hidden_states, gamma, scrm_coal_times):
    st = np.arange(len(hidden_states) - 1)[:, None]
    ret = []
    for gamma_col, block in zip(gamma.T, grouper(tree_obs_iter(l1, l2, scrm_coal_times), block_size)):
        # scrm coal times are in 4 * N0 units, whereas we are using the 2 * N0 scaling
        m = np.searchsorted(2 * hidden_states, block) - 1
        ret.append(((st - m)**2 * gamma_col[:, None]).mean())
    return ret



