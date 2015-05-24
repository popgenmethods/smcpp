import random
import sys
cimport numpy as np
import numpy as np
import logging
import moran_model
import collections

logger = logging.getLogger(__name__)

# cdef void pylogger(const string &msg, const string& lvl):
    # loglvl = logging.getattr(lvl)
    # loglvl(msg)

# cdef class Demography:
#     cdef PiecewiseExponential *pexp
#     cdef vector[double] _logu, _logv, _logs
#     cdef int _K
#     def __init__(self, logu, logv, logs):
#         assert len(logu) == len(logv) == len(logs)
#         self._logu = logu
#         self._logv = logv
#         self._logs = logs
#         self.pexp = new PiecewiseExponential(self._logu, self._logv, self._logs)
#         self._K = len(logu)
# 
#     @property
#     def K(self):
#         return self._K
# 
#     @property
#     def logu(self):
#         return self._logu
# 
#     @property
#     def logv(self):
#         return self._logv
# 
#     @property
#     def logs(self):
#         return self._logs
# 
#     def __dealloc__(self):
#         del self.pexp
# 
#     def inverse_rate(self, double y):
#         return self.pexp.double_inverse_rate(y, 0, 1)
# 
#     def print_debug(self):
#         self.pexp.print_debug()
# 
#     def R(self, double y):
#         return self.pexp.double_R(y)
# 
# cdef class PyAdMatrix:
#     cdef AdMatrix A
# 
# cdef class TransitionWrapper:
#     cdef Transition* transition
#     def __dealloc__(self):
#         del self.transition

# Everything needs to be C-order contiguous to pass in as
# flat arrays
aca = np.ascontiguousarray

cdef vector[double] from_list(lst):
    cdef vector[double] ret
    for l in lst:
        ret.push_back(l)
    return ret

def log_likelihood(a, b, s, int n, int S, int M, obs_list, hidden_states, 
        double rho, double theta, double reg_a, double reg_b, double reg_s,
        int numthreads, seed=None, viterbi=False, jacobian=False):
    # Create stuff needed for computation
    # Sample conditionally; populate the interpolating rate matrices
    assert len(a) == len(b) == len(s)
    K = len(a)
    mats, ts = moran_model.interpolators(n)
    if not seed:
        seed = np.random.randint(0, sys.maxint)
    set_seed(seed)
    cdef vector[double*] expM
    cdef double[:, :, ::1] mmats = aca(mats)
    cdef int i
    for i in range(mats.shape[0]):
        expM.push_back(&mmats[i, 0, 0])
    cdef double[:, ::1] mjac
    jac = aca(np.zeros((3, K)))
    mjac = jac
    cdef int[:, ::1] mobs
    cdef vector[int*] vobs
    cdef int L = obs_list[0].shape[0]
    contig_obs = [aca(ob, dtype=np.int32) for ob in obs_list]
    for ob in contig_obs:
        assert ob.shape == (L, 3)
        mobs = ob
        vobs.push_back(&mobs[0, 0])
    cdef vector[vector[int]] viterbi_paths
    cdef adouble ad

    if jacobian:
        ad = loglik[adouble](
                from_list(a), from_list(b), from_list(s), 
                n, 
                S, M, 
                from_list(ts), expM, 
                L, vobs, 
                from_list(hidden_states), rho, theta, 
                numthreads,
                viterbi, viterbi_paths, 
                reg_a, reg_b, reg_s)
        fill_jacobian(ad, &mjac[0, 0])
        ret = (toDouble(ad), jac)
        if viterbi:
            ret += (np.sum(viterbi_paths, axis=0),)
    else:
        ret = loglik[double](
            from_list(a), from_list(b), from_list(s), 
            n, 
            S, M, 
            from_list(ts), expM, 
            L, vobs, 
            from_list(hidden_states), rho, theta, 
            numthreads,
            viterbi, viterbi_paths, 
            reg_a, reg_b, reg_s)
    return ret

def _reduced_sfs(sfs):
    n = sfs.shape[1] - 1
    reduced_sfs = np.zeros(n + 1)
    for i in range(3):
        for j in range(n + 1):
            if i + j < n + 1:
                reduced_sfs[i + j] += sfs[i][j]
    return reduced_sfs

def sfs(a, b, s, int n, int S, int M, double tau1, double tau2, int numthreads, double theta, seed=None, jacobian=False):
    assert len(a) == len(b) == len(s)
    K = len(a)
    mats, ts = moran_model.interpolators(n)
    if not seed:
        seed = np.random.randint(0, sys.maxint)
    set_seed(seed)
    cdef vector[double*] expM
    cdef double[:, :, ::1] mmats = aca(mats)
    cdef int i
    for i in range(mats.shape[0]):
        expM.push_back(&mmats[i, 0, 0])
    sfs = aca(np.zeros([3, n + 1]))
    cdef double[:, ::1] msfs = sfs
    cdef double[:, :, :, ::1] mjac 
    if jacobian:
        jac = aca(np.zeros((3, n + 1, 3, K)))
        mjac = jac
        cython_calculate_sfs_jac(a, b, s, n, S, M, ts, expM, tau1, tau2, numthreads, theta, &msfs[0, 0], &mjac[0, 0, 0, 0])
        return (sfs, _reduced_sfs(sfs), jac)
    else:
        cython_calculate_sfs(a, b, s, n, S, M, ts, expM, tau1, tau2, numthreads, theta, &msfs[0, 0])
        return (sfs, _reduced_sfs(sfs))

def transition(a, b, s, hidden_states, rho, jacobian=False):
    assert hidden_states[0] == 0.0
    assert hidden_states[-1] == np.inf
    M = len(hidden_states) - 1
    trans = aca(np.zeros([M, M]))
    cdef double[:, ::1] mtrans = trans
    cdef double[:, :, :, ::1] mjac
    if jacobian:
        jac = aca(np.zeros((M, M, 3, len(a))))
        mjac = jac
        cython_calculate_transition_jac(a, b, s, hidden_states, rho, &mtrans[0, 0], &mjac[0, 0, 0, 0])
        return (trans, jac)
    else:
        cython_calculate_transition(a, b, s, hidden_states, rho, &mtrans[0, 0])
        return trans

# def hmm(Demography demo, sfs_list, obs_list, hidden_states, rho, theta, numthreads=1, 
#         viterbi=False, double reg_a=1, double reg_b=1, double reg_s=1):
#     print("in hmm")
#     cdef int[:, ::1] mobs
#     cdef vector[int*] vobs
#     L = obs_list[0].shape[0]
#     contig_obs = [aca(ob, dtype=np.int32) for ob in obs_list]
#     for ob in contig_obs:
#         assert ob.shape == (L, 3)
#         mobs = ob
#         vobs.push_back(&mobs[0, 0])
#     cdef vector[AdMatrix] emission
#     cdef PyAdMatrix wrap
#     for sfs in sfs_list:
#         wrap = sfs
#         emission.push_back(wrap.A)
#     logger.debug("Computing HMM likelihood for %d data sets using %d threads" % (len(obs_list), numthreads))
#     jac = aca(np.zeros((3, demo.K)))
#     cdef double[:, ::1] mjac = jac
#     cdef vector[vector[int]] paths
#     cdef double logp = compute_hmm_likelihood(&mjac[0, 0], demo.pexp[0], 
#             emission, L, vobs, hidden_states, rho, 
#             numthreads, paths, viterbi, reg_a, reg_b, reg_s)
#     ret = (logp, jac)
#     if viterbi:
#         ret += (np.sum(paths, axis=0),)
#     return ret
