from __future__ import absolute_import, division, print_function
import momi, momi.demography
import scipy.linalg, scipy.stats
import numpy as np
import multiprocessing as mp
from collections import namedtuple

from . import model, logging, _smcpp, util
from .moran_eigensystem import MoranEigensystem

import time

class PiecewiseModel(object):
    def __init__(self, s, a, dlist):
        self.s = s
        self.a = a
        self.dlist = dlist
    def stepwise_values(self):
        return self.a

class Timer:    
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        print("took %g" % (self.end - self.start,))

logger = logging.getLogger(__name__)

class JointCSFS(object):
    def __init__(self, n1, n2, a1, a2, hidden_states):
        '''
        Return JCSFS for model1 and model2 which split at time split --
        more recently than split, model 1's size is followed.

        n1 and n2 are the total (undistinguished, haploid) sample sizes
        in pops 1 and 2. a1 and a2 are the number of distinguished
        lineages sampled in each. Either a1 = 2, so that the
        distinguished individual came from pop. 1, or a1 = a2 = 1, in
        which case the distinguished lineages are sampled from different
        populations.
        '''
        self._K = 10
        self._n1 = n1
        self._n2 = n2
        self._a1 = a1
        self._a2 = a2
        self._hs = hidden_states
        self._jcsfs = np.zeros([
            len(hidden_states) - 1, self._a1 +
            1, self._n1 + 1, self._a2 + 1, self._n2 + 1
        ])
        self._hs_pairs = list(zip(hidden_states[:-1], hidden_states[1:]))
        self._Mn1n20 = MoranEigensystem(n1 + n2, 0)
        self._Mn10 = MoranEigensystem(n1, 0)
        self._Mn11 = MoranEigensystem(n1, 1)
        self._Mn2 = MoranEigensystem(n2)

        # Create a pool of workers to speed up computations below.
        # *Very* important to use "spawn" here and not "fork". The
        # latter has big problems with multithreading in _smcpp. I kept
        # getting deadlocks.
        self._pool = mp.get_context("spawn").Pool()

        # For now we only support a2=0.
        assert a1 + a2 == 2
        assert 0 <= a1 <= 2
        assert 0 <= a2 <= 1
        assert a2 == 0, "distinguished lineages originating in different populations not currently supported"

    def compute(self, model1, model2, split):
        Rts1, Rts2 = [_R(m, split) for m in [model1, model2]]
        args = (split, model1, model2, Rts1, Rts2) + \
            tuple([getattr(self, "_" + s)
                   for s in "a1 a2 n1 n2 Mn1n20 Mn10 Mn11 Mn2 K".split()])
        mapfun = self._pool.map
        mapfun = map
        self._jcsfs[:] = list(mapfun(_parallel_helper,
            [_ParallelArgs(t1, t2, *args) for t1, t2 in self._hs_pairs]))
        assert np.all(np.isfinite(self._jcsfs))
        return self._jcsfs

_ParallelArgs = namedtuple("_ParallelArgs",
                           "t1 t2 split model1 model2 Rts1 Rts2 "
                           "a1 a2 n1 n2 Mn1n20 Mn10 Mn11 Mn2 K")

def _parallel_helper(pargs):
    t1, t2, split = pargs[:3]
    if t1 < t2 <= split:
        ret = _jcsfs_helper_tau_below_split(pargs)
    elif split <= t1 < t2:
        ret = _jcsfs_helper_tau_above_split(pargs)
    else:
        assert t1 < split < t2
        r1 = _jcsfs_helper_tau_below_split(pargs._replace(t2=split))
        r2 = _jcsfs_helper_tau_above_split(pargs._replace(t1=split))
        eR1t1, eR1t2, eRts = [np.exp(-_R(pargs.model1, t)) for t in (t1, t2, split)]
        w = (eRts - eR1t2) / (eR1t1 - eR1t2)
        assert 0 <= w <= 1
        ret = r1 * (1. - w) + r2 * w
    assert np.all(np.isfinite(ret))
    return ret


def _jcsfs_helper_tau_above_split(pargs):
    '''JCSFS conditional on distinguished lineages coalescing at split <= t1 < t2'''
    t1, t2, split, model1, model2, Rts1, Rts2, a1, a2, \
            n1, n2, Mn1n20, Mn10, Mn11, Mn2, K = pargs
    assert split <= t1 < t2
    assert a1 == 2
    ret = np.zeros([a1 + 1, n1 + 1, a2 + 1, n2 + 1])
    # To compute this we calculate the CSFS at split time and then propagate forward
    # by hypergeometric sampling and Moran model (a la momi)
    s = model1.s
    a = model1.stepwise_values()
    cs = _cumsum0(s)
    i = np.searchsorted(cs, split)
    cs[i - 1] = split
    sp = np.diff(cs[i - 1:])
    ap = a[i - 1:]
    shim_model = PiecewiseModel(sp, ap, [])
    # CSFS with time "shifted back" by split units
    rsfs = _smcpp.raw_sfs(shim_model, n1 + n2, t1 - split, t2 - split).astype('float')
    eMn10 = Mn10.expm(float(Rts1))
    eMn11 = Mn11.expm(float(Rts1))
    eMn12 = eMn10[::-1, ::-1]
    eMn2 = Mn2.expm(float(Rts2))
    # Moran transition in each subpopulation from split to present
    for b1 in range(n1 + 1):
        for b2 in range(n2 + 1):
            for nseg in range(n1 + n2 + 1):
                # pop1 has to get at least nseg - n2 derived alleles
                for np1 in range(max(nseg - n2, 0), min(nseg, n1) + 1):
                    np2 = nseg - np1
                    h = scipy.stats.hypergeom.pmf(np1, n1 + n2, nseg, n1) 
                    ret[0, b1, 0, b2] += (
                            h * rsfs[0, nseg] *
                            eMn10[np1, b1] * eMn2[np2, b2]
                            )
                    ret[1, b1, 0, b2] += (
                            h * rsfs[1, nseg] *
                            eMn11[np1, b1] * eMn2[np2, b2]
                            )
                    ret[2, b1, 0, b2] += (
                            h * rsfs[2, nseg] *
                            eMn12[np1, b1] * eMn2[np2, b2]
                            )
    # to add in the time beneath split, we fudge slightly and just compute
    # another csfs conditioned on coalescence in (split, split + epsilon).
    rsfs_below = _smcpp.raw_sfs(model1, n1, split - 1e-6, split + 1e-6, True).astype('float')
    ret[:, :, 0, 0] += rsfs_below
    # Since population 2 has no distinguished lineages, we get this by
    # marginalizing over the csfs.
    if n2 > 0:
        if n2 == 1:
            ret[0, 0, 0, 1] += split
        else:
            # n2 >= 2
            rsfs_below = _smcpp.raw_sfs(model2, n2 - 2, split - 1e-6, split + 1e-6).astype('float')
            rsfs_below = util.undistinguished_sfs(rsfs_below, False)
            aoeu
            ret[0, 0, 0] += rsfs_below
    assert(np.all(ret >= 0))
    return ret

def _jcsfs_helper_tau_below_split(pargs):
    '''JCSFS conditional on distinguished lineages coalescing at t1 < t2 <= split'''
    t1, t2, split, model1, model2, Rts1, Rts2, a1, a2, \
            n1, n2, Mn1n20, Mn10, Mn11, Mn2, K = pargs
    assert t1 < t2 <= split
    assert a1 == 2
    ret = np.zeros([a1 + 1, n1 + 1, a2 + 1, n2 + 1])
    S2 = np.diag(np.arange(1, n1 + 1, dtype=float)) / (n1 + 1)
    S0 = np.eye(n1) - S2
    rsfs = _smcpp.raw_sfs(model1, n1, t1, t2, True)
    ret[:, :, 0, 0] = rsfs
    m2s = _cumsum0(model2.s)
    tm2 = np.searchsorted(m2s, split)
    m2s = m2s[:tm2]
    m2a = model2.stepwise_values()[:tm2].astype('float')
    p2_events = [("-en", tt, "pop2", aa) for tt, aa in zip(m2s, m2a)]
    configs = [((b1, n1 + 1 - b1), (b2, n2 - b2))
               for b1 in range(n1 + 2)
               for b2 in range(n2 + 1)
               if 0 < b1 + b2 < n1 + n2 + 1]
    eta = _smcpp.PyRateFunction(model1, [])
    for t, Rt in eta.random_coal_times(t1, t2, K):
        # Compute jsfs using momi
        m1s = _cumsum0(model1.s)
        tm1 = np.searchsorted(m1s, t) - 1
        m1s[tm1] = t
        m1s = m1s[tm1:]
        a = model1.stepwise_values()[tm1:].astype('float')
        events = _model_to_momi_events(m1s, a, "pop1")
        events.append(("-ej", split, "pop2", "pop1"))
        events += p2_events
        demo = momi.demography.make_demography(events, ["pop1", "pop2"], 
                (n1 + 1, n2), sampled_t=[float(t), 0.])
        mconf = momi.config_array(("pop1","pop2"), configs)
        sfs = np.zeros([n1 + 2, n2 + 1])
        esfs = momi.expected_sfs(demo, mconf, mut_rate=1.0)
        sfs.flat[1:-1] = esfs
        trans0 = Mn10.expm(float(Rt))
        trans2 = trans0[::-1, ::-1]
        ret[0, 0, 0] += sfs[0] / K
        ret[2, n1, 0] += sfs[-1] / K
        ret[0, :, 0] += sfs[1:-1].T.dot(S0).dot(trans0[1:]).T / K
        ret[2, :, 0] += sfs[1:-1].T.dot(S2).dot(trans2[:-1]).T / K
        assert(np.all(ret >= 0))
    return ret

def _cumsum0(ary):
    return np.concatenate([[0], np.cumsum(ary)])

def _moran_rate_matrix(N):
    ret = np.zeros([N + 1, N + 1])
    i, j = np.indices(ret.shape)
    # Superdiagonal
    k = np.arange(N)
    ret[j == i + 1] = 0.5 * k * (N - k)
    # Subdiagonal
    k += 1
    ret[j == i - 1] = 0.5 * k * (N - k)
    ret[i == j] = -np.sum(ret, axis=1)
    return ret

def _modified_rate_matrix(N, a):
    ret = np.zeros([N + 1, N + 1])
    i, j = np.indices(ret.shape)
    # Superdiagonal
    k = np.arange(N)
    ret[j == i + 1] = a * (N - k) + 0.5 * k * (N - k)
    # Subdiagonal
    k += 1
    ret[j == i - 1] = (2 - a) * k + 0.5 * k * (N - k)
    ret[i == j] = -np.sum(ret, axis=1)
    return ret

def _R(model, t):
    # integral of cumulate rate function.
    return _smcpp.PyRateFunction(model, []).R(t)

def _model_to_momi_events(s, a, pop):
    return [("-en", tt, pop, aa) for tt, aa in zip(s, a)]
