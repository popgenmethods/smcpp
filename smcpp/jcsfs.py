import momi
import scipy.linalg

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
        assert 0 <= a2 <= 1 0

    def _jcsfs_helper_tau_above_split(tau, n1, n2, a1, a2, model1, model2, split):
        '''JCSFS conditional on '''
        if tau < split:
            pass

    def _jcsfs_helper_tau_below_split(self, t1, t2):
        '''JCSFS conditional on distinguished lineages coalescing at t1 < t2 <= split'''
        assert t < self._split
        assert a1 == 2
        ret = np.zeros([3, self.n1 + 1, 3, self.n2 + 1])
        ret[:, :, 0, 0] = _smcpp.raw_sfs_below(self._model1, t1, t2)
        for t in _smcpp.random_coal_times(self._model1):
            # Compute jsfs using momi
