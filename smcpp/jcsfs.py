# This computes the joint CSFS. This implemenation proved too slow so it
# was later re-implemented in C++; see src/jcsfs.cpp. This file remains
# for illustrative purposes since it is better documented than the C++
# version.

from __future__ import absolute_import, division, print_function
import scipy.linalg, scipy.stats
import numpy as np
from collections import namedtuple
import logging
import time

from . import model, _smcpp, util
from .moran_eigensystem import MoranEigensystem

class Timer:    
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        print("took %g" % (self.end - self.start,))

logger = logging.getLogger(__name__)

class JointCSFS(object):
    def __init__(self, n1, n2, a1, a2, hidden_states, K=10):
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
        self._K = K
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
        self._Mn10 = MoranEigensystem(n1, 0)
        self._Mn11 = MoranEigensystem(n1, 1)
        self._Mn12 = MoranEigensystem(n1, 2)
        self._Mn1 = MoranEigensystem(n1 + 1)
        self._Mn2 = MoranEigensystem(n2)

        # Create a pool of workers to speed up computations below.
        # *Very* important to use "spawn" here and not "fork". The
        # latter has big problems with multithreading in _smcpp. I kept
        # getting deadlocks.
        # self._pool = mp.Pool()

        # For now we only support a2=0.
        assert a1 + a2 == 2
        assert 0 <= a1 <= 2
        assert 0 <= a2 <= 1
        assert a2 == 0, "distinguished lineages originating in different populations not currently supported"

    def compute(self, model1, model2, split):
        Rts1, Rts2 = [float(_R(m, split)) for m in [model1, model2]]
        args = (split, model1, model2, Rts1, Rts2) + \
            tuple([getattr(self, "_" + s)
                   for s in "a1 a2 n1 n2 Mn10 Mn11 Mn12 Mn1 Mn2 K".split()])
        mapfun = map
        self._jcsfs[:] = list(mapfun(_parallel_helper,
            [_ParallelArgs(t1, t2, *args) for t1, t2 in self._hs_pairs]))
        assert np.all(np.isfinite(self._jcsfs))
        return self._jcsfs

_ParallelArgs = namedtuple("_ParallelArgs",
                           "t1 t2 split model1 model2 Rts1 Rts2 "
                           "a1 a2 n1 n2 Mn10 Mn11 Mn12 Mn1 Mn2 K")

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
        eR1t1, eR1t2, eRts = [np.exp(-float(_R(pargs.model1, t))) for t in (t1, t2, split)]
        w = (eRts - eR1t2) / (eR1t1 - eR1t2)
        assert 0 <= w <= 1
        ret = r1 * (1. - w) + r2 * w
    # Since population 2 has no distinguished lineages, we get this by
    # marginalizing over the csfs.
    n2 = pargs.n2
    model2 = pargs.model2
    if n2 > 0:
        if n2 == 1:
            ret[0, 0, 0, 1] += split
        else:
            # n2 >= 2
            # Compute the truncated sfs by getting the truncated sfs
            # conditioned on distinguished lineages coalescing in (0,
            # inf) and the marginalizing over those two lineages.
            tsfs_below = _truncated_csfs(model2, n2 - 2, split, 0., np.inf)
            rsfs_below = util.undistinguished_sfs(tsfs_below)[1:].astype('float')
            ret[0, 0, 0, 1:-1] += rsfs_below
            ret[0, 0, 0, -1] += split - np.arange(1, n2).dot(rsfs_below) / n2
    assert np.all(np.isfinite(ret))
    ret[0, 0, 0, 0] = ret[-1, -1, -1, -1] = 0. # set nonsegregating sites to zero
    return ret


def _jcsfs_helper_tau_above_split(pargs):
    '''JCSFS conditional on distinguished lineages coalescing at split <= t1 < t2'''
    t1, t2, split, model1, model2, Rts1, Rts2, a1, a2, \
            n1, n2, Mn10, Mn11, Mn12, Mn1, Mn2, K = pargs
    assert split <= t1 < t2
    assert a1 == 2
    ret = np.zeros([a1 + 1, n1 + 1, a2 + 1, n2 + 1])
    # To compute this we calculate the CSFS at split time and then propagate forward
    # by hypergeometric sampling and Moran model (a la momi)
    shifted_model = _shift_model(model1, split)
    # CSFS with time "shifted back" by split units
    rsfs = _smcpp.raw_sfs(shifted_model, n1 + n2, t1 - split, t2 - split).astype('float')
    eMn1 = [Mn10.expm(float(Rts1)), Mn11.expm(float(Rts1))]
    eMn1.append(eMn1[0][::-1, ::-1])
    eMn1 = np.array(eMn1)
    eMn2 = Mn2.expm(float(Rts2))
    # Moran transition in each subpopulation from split to present
    for b1 in range(n1 + 1):
        for b2 in range(n2 + 1):
            for nseg in range(n1 + n2 + 1):
                # pop1 has to get at least nseg - n2 derived alleles
                for np1 in range(max(nseg - n2, 0), min(nseg, n1) + 1):
                    np2 = nseg - np1
                    h = scipy.stats.hypergeom.pmf(np1, n1 + n2, nseg, n1) 
                    for i in range(3):
                        ret[i, b1, 0, b2] += (
                            h * rsfs[i, nseg] *
                            eMn1[i, np1, b1] * 
                            eMn2[np2, b2]
                        )
    # to add in the time beneath split, we fudge slightly and just compute
    # another csfs_below conditioned on coalescence in (split +- epsilon).
    rsfs_below = _smcpp.raw_sfs(model1, n1, split - 1e-6, split + 1e-6, True).astype('float')
    ret[:, :, 0, 0] += rsfs_below
    assert(np.all(ret >= -1e-10))
    ret = np.maximum(0., ret)
    return ret

def _jcsfs_helper_tau_below_split(pargs):
    '''JCSFS conditional on distinguished lineages coalescing at t1 < t2 <= split'''
    t1, t2, split, model1, model2, Rts1, Rts2, a1, a2, \
            n1, n2, Mn10, Mn11, Mn12, Mn1, Mn2, K = pargs
    assert t1 < t2 <= split
    assert a1 == 2
    ret = np.zeros([a1 + 1, n1 + 1, a2 + 1, n2 + 1])
    # Below the split, we compute the "truncated csfs" by crashing the
    # population above the split.
    ret[:, :, 0, 0] = _truncated_csfs(model1, n1, split, t1, t2)
    ret[2, n1, 0, 0] = split - \
            np.arange(1, n1 + 2).dot(util.undistinguished_sfs(ret[:, :, 0, 0])[1:]) / (n1 + 2)

    # Above the split, we compute the regular SFS on n1 + n2 + 1
    # lineages (by marginalizing the CSFS on n1 + n2 - 1) and then moran
    # down.
    shifted_model1 = _shift_model(model1, split)
    shifted_csfs = _smcpp.raw_sfs(shifted_model1, n1 + n2 - 1, 0., np.inf)
    sfs_above_split = util.undistinguished_sfs(shifted_csfs)[1:]
    # transition matrices 
    # Compute "averaged transition matrices"
    eta = _smcpp.PyRateFunction(model1, [])
    eMn10_avg = np.zeros([n1 + 2, n1 + 1], dtype=object)
    eMn12_avg = np.zeros_like(eMn10_avg)
    eMn2 = Mn2.expm(Rts2)
    S2 = np.arange(n1 + 2, dtype=float) / (n1 + 1)
    S2 = S2[None, :]
    S0 = 1. - S2
    for t, Rt in eta.random_coal_times(t1, t2, K):
        t = float(t)
        Rt = float(Rt)
        A = Mn1.expm(Rts1 - Rt) # normal Moran dynamics down time t
        # Then either one lineage splits off into 2 distinguished
        # with prob S2
        B = Mn10.expm(Rt)
        eMn10_avg += (A * S0)[:, :-1].dot(B)
        # Or it doesn't
        # Process eMn10 is reverse of eMn12
        C = Mn12.expm(Rt)
        # B[::-1, ::-1]
        eMn12_avg += (A * S2)[:, 1:].dot(C)
    eMn10_avg /= K
    eMn12_avg /= K
    # Now moran down
    for b1 in range(n1 + 1): # number of derived alleles in pop1
        for b2 in range(n2 + 1): # number of derived alleles in pop2
            for nseg in range(1, n1 + n2 + 1):
                # pop1 has to get at least nseg - n2 derived alleles
                for np1 in range(max(nseg - n2, 0), min(nseg, n1 + 1) + 1):
                    np2 = nseg - np1
                    h = scipy.stats.hypergeom.pmf(np1, n1 + n2 + 1, nseg, n1 + 1) 
                    ret[0, b1, 0, b2] += (
                            h * sfs_above_split[nseg - 1] *
                            eMn10_avg[np1, b1] * eMn2[np2, b2]
                            )
                    ret[2, b1, 0, b2] += (
                            h * sfs_above_split[nseg - 1] *
                            eMn12_avg[np1, b1] * eMn2[np2, b2]
                            )
    assert np.all(ret >= -1e-10)
    ret = np.maximum(0., ret)
    return ret

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
    if np.isinf(t):
        return np.inf
    return _smcpp.PyRateFunction(model, []).R(t)

def _model_to_momi_events(s, a, pop):
    return [("-en", tt, pop, aa) for tt, aa in zip(s, a)]

def _truncated_csfs(_model, n, split, t1, t2):
    assert n >= 0
    s = _model.s
    a = _model.stepwise_values()
    cs = np.cumsum(s)
    cs[-1] = np.inf
    ind = np.searchsorted(cs, split)
    cs = np.insert(cs, ind, split)
    sp = np.diff(np.concatenate([[.0], cs]))[:ind + 1]
    ap = np.append(a[:ind + 1], 1e-8) # create artificial crash to truncate frequency spectrum
    sp = np.append(sp, 1.)
    assert len(sp) == len(ap)
    _model = model.PiecewiseModel(sp, ap, _model.dlist)
    return _smcpp.raw_sfs(_model, n, t1, t2)

def _shift_model(_model, shift):
    'Shift the model back :shift: units in time.'
    s = _model.s
    a = _model.stepwise_values()
    cs = util.cumsum0(s)
    cs[-1] = np.inf
    i = np.searchsorted(cs, shift)
    cs[i - 1] = shift
    sp = np.diff(cs[i - 1:])
    sp[-1] = 1.0
    ap = a[i - 1:]
    return model.PiecewiseModel(sp, ap, _model.dlist)
