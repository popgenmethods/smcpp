import numpy as np
import logging

import _pypsmcpp
from transition import TransitionMatrix

# This class wires together the transition and sfs calculations, plus jacobians.
class Demography:
    '''Class representing a piecewise-exponential demography:

        eta(t) = a_k * exp(b_k * (t - t_{k - 1})), t_{k - 1} <= t < t_k.

    To enforce positivity constraints later on, we pass in sqrt(a_k) and sqrt(s_k).
    We then use a_k := sqrt(a_k)**2 and t_k = s_1**2 + ... + s_k**2.
    '''

    def __init__(self, sqrt_a, b, sqrt_s, hidden_states):
        self.sqrt_a = np.array(sqrt_a)
        self.b = np.array(b)
        self.sqrt_s = np.array(sqrt_s)
        assert self.sqrt_a.shape == self.b.shape == self.sqrt_s.shape
        assert len(self.sqrt_a.shape) == 1
        self.K = self.sqrt_a.shape[0]
        self.hs = np.array(hidden_states)
        assert self.hs[-1] == np.inf
        assert self.hs[0] == 0
        self.M = self.hs.shape[0] - 1
        self.eta = _pypsmcpp.PiecewiseExponentialWrapper(sqrt_a, b, sqrt_s)

    def transition(self, rho):
        tm = TransitionMatrix(rho, self.R, self.hs)
        jac = np.zeros((3, self.K) + tm.A.shape, dtype=object)
        for i, obj in enumerate([self.ad_sqrt_a, self.ad_b, self.ad_sqrt_s]):
            for m in range(self.M):
                jac[i, :, m] = np.transpose(ad.jacobian(tm.A[m], obj))
        return (tm.A, jac)
        

    def sfs(self, n, M, pool=None):
        '''Compute (simulate) sfs for n individuals using M simulations. If pool != None,
        use a pool of multiprocessing instances to run the simulations in parallel.'''
        if pool is not None:
            k = pool._processes
            M //= k
            logging.info("Multiprocessing for SFS computation enabled (%d workers)" % k)
            mapfun = pool.map
        else:
            k = 1
            logging.info("Multiprocessing for SFS computation disabled")
            mapfun = map
        ret = np.zeros([self.hs.shape[0] - 1, 3, n])
        ret_jac = np.zeros([3, self.K, self.hs.shape[0] - 1, 3, n])
        for i, (tau1, tau2) in enumerate(zip(self.hs[:-1], self.hs[1:])):
            logging.info("Computing sfs for interval [%f, %f)" % (tau1, tau2))
            args = [(M, self.sqrt_a, self.b, self.sqrt_s, n, tau1, tau2, theta)] * k
            for _, sfs, jac in mapfun(_call_pypsmcpp, args):
                ret[i] += sfs / k
                ret_jac[:, :, i] += jac / k
        return (ret, ret_jac)


    def pi(self):
        '''Given a vector of hidden states 0 = hs[0] < hs[1] < ... < hs[M] = inf,
        compute probabilities P(1, 2 coalesce in [hs[k - 1], hs[k])) for
        k \in {1, ..., M}.
        '''
        Rinv = self.R.inverse(self.hs[:-1])
        inv = np.exp(-Rinv)
        inv = np.append(inv, [0])
        ret = inv[:-1] - inv[1:]
        jac = np.zeros([3, self.K, self.M])
        for i, obj in enumerate([self.ad_sqrt_a, self.ad_b, self.ad_sqrt_s]):
            jac[i] = np.transpose(ad.jacobian(ret, obj))
        assert abs(ret.sum() - 1) < 1e-6
        return (ret, jac)
