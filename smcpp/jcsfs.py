import momi
import scipy.linalg
import numpy as np

import _smcpp

class JointCSFS(object):
    def __init__(self, n1, n2, a1, a2, model1, model2, split):
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
        self._n1 = n1
        self._n2 = n2
        self._a1 = a1
        self._a2 = a2
        self._model1 = model1
        self._model2 = model2
        self._split = split
        assert a1 + a2 == 2
        assert 0 <= a1 <= 2
        assert 0 <= a2 <= 1
        self._S0 = np.diag(np.arange(1, n1 + 1)) / (n1 + 1)
        self._S2 = np.eye(n1) - self._S0
        self._M0 = _modified_rate_matrix(n1, 0)
        self._M2 = _modified_rate_matrix(n1, 2)

    def _jcsfs_helper_tau_above_split(tau, n1, n2, a1, a2, model1, model2, split):
        '''JCSFS conditional on '''
        if tau < split:
            pass

    def _jcsfs_helper_tau_below_split(self, t1, t2):
        '''JCSFS conditional on distinguished lineages coalescing at t1 < t2 <= split'''
        assert t1 < t2 <= self._split
        assert self._a1 == 2
        ret = np.zeros([3, self._n1 + 1, 3, self._n2 + 1])
        # ret[:, :, 0, 0] = _smcpp.raw_sfs_below(self._model1, t1, t2)
        m2s = _cumsum0(self._model2.s)
        tm2 = np.searchsorted(m2s, self._split)
        m2s[tm2] = self._split
        m2s = m2s[:tm2 + 1]
        m2a = self._model2.stepwise_values()[:tm2 + 1]
        p2_events = [("-en", tt, "pop2", aa) for tt, aa in zip(m2s, m2a)]
        N = 10
        for t, Rt in _smcpp.random_coal_times(self._model1, N):
            # Compute jsfs using momi
            m1s = _cumsum0(self._model1.s)
            tm1 = np.searchsorted(m1s, t) - 1
            m1s[tm1] = t
            m1s = m2s[tm1:]
            a = self._model1.stepwise_values()[tm1:]
            events = [("-en", tt, "pop1", aa) for tt, aa in zip(s, a)]
            events.append(("-ej", self._split, "pop2", "pop1"))
            events += p2_events
            demo = momi.Demography(events, sampled_n=(self._n1 + 1, self._n2), sampled_t=[t, 0.])
            tbl = momi.expected_total_branch_len(demo)
            # For each entry in JCSFS, multiply by corresponding
            # probability of subtending each sample entry.
            tbl

def _cumsum0(ary):
    return np.concatenate([[0], np.cumsum(ary)])

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
