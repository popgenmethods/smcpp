from __future__ import absolute_import, division, print_function
import momi, momi.demography
import scipy.linalg, scipy.stats
import numpy as np
import multiprocessing as mp
import ctypes

from . import model, logging, _smcpp, pool

logger = logging.getLogger(__name__)

class JointCSFS(object):
    def __init__(self, n1, n2, a1, a2, model1, model2, split, hidden_states):
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
        self._model1 = model1
        self._model2 = model2
        self._split = split
        self._hs = hidden_states
        ## FIXME this is hacky. should be replaced by calls to eta, probably
        self._Rts1 = float(_R(self._model1, split))
        self._Rts2 = float(_R(self._model2, split))
        self._jcsfs = np.zeros([len(hidden_states) - 1, self._a1 + 1, self._n1 + 1, self._a2 + 1, self._n2 + 1], dtype=float)
        self._hs_pairs = zip(hidden_states[:-1], hidden_states[1:])
        for t1, t2 in self._hs_pairs:
            assert (t1 < t2 <= split) or (split <= t1 < t2)
        assert a1 + a2 == 2
        assert 0 <= a1 <= 2
        assert 0 <= a2 <= 1
        assert a2 == 0, "distinguished lineages originating in different populations not currently supported"

    def compute(self):
        args = tuple([getattr(self, "_" + x) for x in "split, a1, a2, n1, n2, model1, model2, Rts1, Rts2, K".split(", ")])
        processes = []
        ary = mp.Array('d', self._jcsfs.size)
        for i, (t1, t2) in enumerate(self._hs_pairs):
            processes.append(mp.Process(target=_parallel_helper, args=(ary, self._jcsfs.shape, i, t1, t2) + args))
            processes[-1].start()
        for p in processes:
            p.join()
        # self._jcsfs[:] = list(p.imap(_parallel_helper, ((t1, t2) + args for t1, t2 in self._hs_pairs), 1))
        return self._jcsfs

def _parallel_helper(ary, shape, i, t1, t2, split, *args):
    jcsfs = np.ctypeslib.as_array(ary.get_obj())
    jcsfs.shape = shape
    print(t1, t2)
    if t1 < t2 <= split:
        jcsfs[i] = _jcsfs_helper_tau_below_split(t1, t2, split, *args)
    elif split <= t1 < t2:
        jcsfs[i] = _jcsfs_helper_tau_above_split(t1, t2, split, *args)
    return


def _jcsfs_helper_tau_above_split(t1, t2, split, a1, a2, n1, n2, model1, model2, Rts1, Rts2, K):
    '''JCSFS conditional on distinguished lineages coalescing at t1 < t2 <= split'''
    assert split <= t1 < t2
    assert a1 == 2
    ret = np.zeros([a1 + 1, n1 + 1, a2 + 1, n2 + 1])
    # Two parts: branch length between split and tau, and branch
    # length above tau.
    #
    s = np.array(model1.s, dtype=float)
    a = np.array(model1.stepwise_values(), dtype=float)
    cs = _cumsum0(s)
    i = np.searchsorted(cs, split)
    cs[i - 1] = split
    sp = np.diff(cs[i - 1:])
    ap = a[i - 1:]
    shim_model = model.PiecewiseModel(sp, ap)
    rsfs = _smcpp.raw_sfs(shim_model, n1 + n2 + 2, t1, t2, True)
    M0 = _modified_rate_matrix(n1 + n2, 0)
    S2 = np.arange(1, n1 + n2 + 1, dtype=float) / (n1 + n2 + 1)
    S0 = 1. - S2
    M10 = _modified_rate_matrix(n1, 0)
    M11 = _modified_rate_matrix(n1, 1)
    M21 = _modified_rate_matrix(n1, 2)
    eM10 = scipy.linalg.expm(Rts1 * M10)
    eM11 = scipy.linalg.expm(Rts1 * M11)
    eM21 = scipy.linalg.expm(Rts1 * M21)
    M2 = _moran_rate_matrix(n2)
    eM2 = scipy.linalg.expm(Rts2 * M2)
    # add additional time above tau by numerical integration
    # sample tau, compute sfs at tau, then moran transition to t_s, 
    configs = [((x, n1 + n2 + 1 - x),) for x in range(1, n1 + n2 + 1)]
    for tau, Rtau in _smcpp.random_coal_times(model1, t1, t2, K):
        logger.debug((t1, t2, tau, Rtau))
        # Compute regular old sfs using momi
        m1s = _cumsum0(model1.s)
        tm1 = np.searchsorted(m1s, tau) - 1
        m1s[tm1] = tau
        m1s = m1s[tm1:]
        a = model1.stepwise_values()[tm1:].astype('float')
        # "shift time back" by t
        m1s -= tau
        events = _model_to_momi_events(m1s, a, "pop1")
        demo = momi.demography.make_demography(events, ["pop1"], (n1 + n2 + 1,))
        mconf = momi.config_array(("pop1",), configs)
        esfs = momi.expected_sfs(demo, mconf, mut_rate=1.0)
        # esfs is esfs at tau. first compute transition down to t_s.
        # there are two cases to consider depending on whether a mutation
        # hits distinguished lineages.
        eM0 = scipy.linalg.expm((Rtau - Rts1) * M0)
        eM2 = eM0[::-1, ::-1]
        sfs0_ts = (esfs * S0).dot(eM0[1:])
        sfs2_ts = (esfs * S2).dot(eM2[:-1])
        rsfs[0] += sfs0_ts / K
        rsfs[2] += sfs2_ts / K

    # Moran transition in each subpopulation from t_S to present
    for b1 in range(n1 + 1):
        for b2 in range(n2 + 1):
            for nseg in range(n1 + n2 + 1):
                # pop1 has to get at least nseg - n2 derived alleles
                for np1 in range(max(nseg - n2, 0), min(nseg, n1) + 1):
                    np2 = nseg - np1
                    h = scipy.stats.hypergeom.pmf(np1, n1 + n2, nseg, n1) 
                    ret[0, b1, 0, b2] += (
                            h * rsfs[0, nseg] *
                            eM10[np1, b1] * eM2[np2, b2]
                            )
                    ret[1, b1, 0, b2] += (
                            h * rsfs[1, nseg] *
                            eM11[np1, b1] * eM2[np2, b2]
                            )
                    ret[2, b1, 0, b2] += (
                            h * rsfs[2, nseg] *
                            eM21[np1, b1] * eM2[np2, b2]
                            )
    return ret

def _jcsfs_helper_tau_below_split(t1, t2, split, a1, a2, n1, n2, model1, model2, Rts1, Rts2, K):
    '''JCSFS conditional on distinguished lineages coalescing at t1 < t2 <= split'''
    assert t1 < t2 <= split
    assert a1 == 2
    ret = np.zeros([a1 + 1, n1 + 1, a2 + 1, n2 + 1])
    S2 = np.diag(np.arange(1, n1 + 1, dtype=float)) / (n1 + 1)
    S0 = np.eye(n1) - S2
    M0 = _modified_rate_matrix(n1, 0)
    rsfs = _smcpp.raw_sfs(model1, n1 + 2, t1, t2, True)
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
    for t, Rt in _smcpp.random_coal_times(model1, t1, t2, K):
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
                (n1 + 1, n2), sampled_t=[t, 0.])
        mconf = momi.config_array(("pop1","pop2"), configs)
        sfs = np.zeros([n1 + 2, n2 + 1])
        esfs = momi.expected_sfs(demo, mconf, mut_rate=1.0)
        sfs.flat[1:-1] = esfs
        trans0 = scipy.linalg.expm(Rt * M0)
        trans2 = trans0[::-1, ::-1]
        ret[0, 0, 0] += sfs[0] / K
        ret[2, n1, 0] += sfs[-1] / K
        ret[0, :, 0] += sfs[1:-1].T.dot(S0).dot(trans0[1:]).T / K
        ret[2, :, 0] += sfs[1:-1].T.dot(S2).dot(trans2[:-1]).T / K
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
    cs = np.concatenate([[0.], np.cumsum(model.s)])
    cs[-1] = np.inf
    i = np.searchsorted(cs, t)
    cs = cs[:i + 1]
    a = model.stepwise_values()[:i]
    cs[-1] = t
    s = np.diff(cs)
    return (s * a).sum()

def _model_to_momi_events(s, a, pop):
    return [("-en", tt, pop, aa) for tt, aa in zip(s, a)]
