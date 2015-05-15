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

cdef class Demography:
    cdef PiecewiseExponential *pexp
    cdef vector[double] _sqrt_a
    cdef vector[double] _b
    cdef vector[double] _sqrt_s
    cdef int K
    def __init__(self, sqrt_a, b, sqrt_s):
        assert len(sqrt_a) == len(b) == len(sqrt_s)
        self._sqrt_a = sqrt_a
        self._b = b
        self._sqrt_s = sqrt_s
        self.pexp = new PiecewiseExponential(sqrt_a, b, sqrt_s)
        self.K = len(sqrt_a)

    @property
    def sqrt_a(self):
        return self._sqrt_a

    @property
    def b(self):
        return self._b

    @property
    def sqrt_s(self):
        return self._sqrt_s

    def __dealloc__(self):
        del self.pexp

    def inverse_rate(self, double y):
        return self.pexp.double_inverse_rate(y, 0, 1)

    def print_debug(self):
        self.pexp.print_debug()

    def R(self, double y):
        return self.pexp.double_R(y)

cdef class PyAdMatrix:
    cdef AdMatrix A

cdef class TransitionWrapper:
    cdef Transition* transition
    def __dealloc__(self):
        del self.transition

# Everything needs to be C-order contiguous to pass in as
# flat arrays
aca = np.ascontiguousarray

def sfs(Demography demo, int S, int M, int n, float tau1, float tau2, double theta, extract_output=False, seed=None, int numthreads=1):
    logger.debug("in sfs")
    # Create stuff needed for computation
    # Sample conditionally; populate the interpolating rate matrices
    mats, ts = moran_model.interpolators(n)
    logger.debug("Constructing ConditionedSFS object")
    cdef double t1, t2
    t1, t2 = [demo.R(x) for x in (tau1, tau2)]
    assert all([not(np.isnan(x)) for x in (t1, t2)])
    if seed:
        np.random.seed(seed)
    cdef vector[double*] expM
    cdef double[:, :, ::1] mmats = aca(mats)
    cdef int i
    for i in range(mats.shape[0]):
        expM.push_back(&mmats[i, 0, 0])
    cdef vector[double] vts = ts
    logging.info("Calculating SFS for interval [%f, %f) using %d threads" % (tau1, tau2, numthreads))
    cdef AdMatrix mat = calculate_sfs(demo.pexp[0], n, S, M, ts, expM, t1, t2, numthreads, theta)
    logging.debug("Done")
    cdef PyAdMatrix ret

    # If we are not extracting output (i.e., for testing purposes)
    # then return a (wrapped) pointer to the csfs object for use
    # later on in the forward algorithm.
    if not extract_output:
        ret = PyAdMatrix()
        ret.A = mat
        return ret

    K = len(demo.sqrt_a)
    sfs = aca(np.zeros((3, n + 1)))
    jac = aca(np.zeros((3, n + 1, 3, K)))
    cdef double[:, ::1] msfs = sfs
    cdef double[:, :, :, ::1] mjac = jac
    store_sfs_results(mat, &msfs[0, 0], &mjac[0, 0, 0, 0])
    reduced_sfs = np.zeros(n + 1)
    for i in range(3):
        for j in range(n + 1):
            if i + j < n + 1:
                reduced_sfs[i + j] += sfs[i][j]
    return (reduced_sfs, sfs, jac)

#def transition(Demography demo, hidden_states, double rho, extract_output=False):
#    cdef vector[double] hs = hidden_states
#    assert hidden_states[0] == 0.0
#    assert hidden_states[-1] == np.inf
#    cdef Transition* trans = new Transition(demo.pexp[0], hs, rho)
#    trans.compute()
#    cdef TransitionWrapper ret
#    if not extract_output:
#        ret = TransitionWrapper()
#        ret.trans = transition
#        return ret
#    M = len(hidden_states)
#    tmat = aca(np.zeros((M - 1, M - 1)))
#    tjac = aca(np.zeros((M - 1, M - 1, 3, demo.K)))
#    cdef double[:, ::1] tmatv = tmat
#    cdef double[:, :, :, ::1] tjacv = tjac
#    trans.store_results(&tmatv[0, 0], &tjacv[0, 0, 0, 0])
#    del trans
#    return (tmat, tjac)

def hmm(Demography demo, sfs_list, obs_list, hidden_states, rho, theta, numthreads=1, 
        viterbi=False, double reg_a=1, double reg_b=1, double reg_s=1):
    print("in hmm")
    cdef int[:, ::1] mobs
    cdef vector[int*] vobs
    L = obs_list[0].shape[0]
    contig_obs = [aca(ob, dtype=np.int32) for ob in obs_list]
    for ob in contig_obs:
        assert ob.shape == (L, 3)
        mobs = ob
        vobs.push_back(&mobs[0, 0])
    cdef vector[AdMatrix] emission
    cdef PyAdMatrix wrap
    for sfs in sfs_list:
        wrap = sfs
        emission.push_back(wrap.A)
    logger.info("Computing HMM likelihood for %d data sets using %d threads" % (len(obs_list), numthreads))
    jac = aca(np.zeros((3, demo.K)))
    cdef double[:, ::1] mjac = jac
    cdef vector[vector[int]] paths
    cdef double logp = compute_hmm_likelihood(&mjac[0, 0], demo.pexp[0], emission, L, vobs, hidden_states, rho, 
            numthreads, paths, viterbi, reg_a, reg_b, reg_s)
    ret = (logp, jac)
    if viterbi:
        c = collections.Counter()
        for i in range(paths.size()):
            c.update(collections.Counter(paths[i]))
        ret += (c,)
    return ret
