from __future__ import division
import numpy as np
import itertools as it
from collections import Counter
import logging
from util import memoize
import shelve
import scipy.sparse.linalg
import scipy.sparse
import scipy.interpolate
import sys

from _expm import expm

logging.basicConfig(level=logging.INFO)

@memoize
def moran_states(N, a):
    return [(a, k) for k in xrange(N + 1)]

def rate_matrix(N, a):
    states = moran_states(N, a)
    L = len(states)
    ij = []
    data = []
    diag = [0.] * (N + 1)
    for k in range(0, N + 1):
        si1 = states.index((a, k))
        if k > 0:
            ij.append([si1, states.index((a, k - 1))])
            v = (2 - a) * k + 0.5 * k * (N - k)
            data.append(v)
            diag[si1] += -v
        if k < N:
            ij.append([si1, states.index((a, k + 1))])
            v = a * (N - k) + 0.5 * k * (N - k)
            data.append(v)
            diag[si1] += -v
    ij += [(i, i) for i in range(N + 1)]
    data += diag
    M = scipy.sparse.coo_matrix((np.array(data), np.array(ij).T), shape=(L, L))
    return M

def p_transition(N, t, s1, s2):
    """Probability of transition s1 to s2 in time t"""
    a1, b1 = s1
    a2, b2 = s2
    # Below tau indistinguished states do not communicate
    if a1 != a2:
        return 0.0
    i1, i2 = [moran_states(N, a1).index(x) for x in (s1, s2)]
    M = rate_matrix(N, a1)

def p_transition_exp(N, a, t):
    Q = rate_matrix(N, a).todense()
    return expm(Q * t)
    # print scipy.sparse.linalg.expm_multiply(t * Q, v)

def interpolators(N):
    _interp_memo = shelve.open(".moran.dat", protocol=2)
    key = str(N)
    if key not in _interp_memo:
        logging.info("Constructing interpolating matrix exponentials (only needs to happen once)")
        Q = rate_matrix(N, 2).todense()
        t = 1e-8
        # for a=2, absorbing state is N
        A = np.zeros((N + 1, N + 1))
        A[:, -1] += 1.0
        M = expm(Q * t)
        bdslow = np.zeros(A.shape)
        bdshigh = np.zeros(A.shape)
        eps = 1.0
        np.set_printoptions(linewidth=100)
        u = []
        x = []
        while eps > 1e-8:
            logging.info("Interpolation progress: %.2f%%" % (1e-6 / eps));
            u.append(t)
            x.append(M)
            t += 0.05
            M = expm(Q * t)
            eps = np.linalg.norm(M - A)
        logging.info("Interpolation progress: Completed")
        u = np.array([0] + u + [np.inf])
        x = np.array([np.eye(N + 1)] + x + [A])
        _interp_memo[key] = (x, u)
    return _interp_memo[key]

if __name__=="__main__":
    interpolators(int(sys.argv[1]))
