cimport numpy as np
import numpy as np
import logging
import moran_model

# cdef void pylogger(const string &msg, const string& lvl):
    # loglvl = logging.getattr(lvl)
    # loglvl(msg)

cdef class PiecewiseExponentialWrapper:
    cdef PiecewiseExponential* pexp
    def __init__(self, sqrt_a, b, sqrt_s):
        assert len(sqrt_a) == len(b) == len(sqrt_s)
        self.pexp = new PiecewiseExponential(sqrt_a, b, sqrt_s)
    def __dealloc__(self):
        del self.pexp

    def inverse_rate(self, double y):
        return self.pexp.double_inverse_rate(y, 0, 1)

    def print_debug(self):
        self.pexp.print_debug()

cdef class ConditionedSFSWrapper:
    cdef ConditionedSFSptr csfs
    def __dealloc__(self):
        del self.csfs

cdef class TransitionWrapper:
    cdef Transition* transition
    def __dealloc__(self):
        del self.transition

# Everything needs to be C-order contiguous to pass in as
# flat arrays
aca = np.ascontiguousarray

def exp1_conditional(double a, double b, size):
    # If X ~ Exp(1), 
    # P(X < x | a <= X <= b) = (e^-a - e^-x) / (e^-a - e^-b)
    # so P^-1(y) = -log(e^-a - (e^-a - e^-b) * y)
    #            = -log(e^-a(1 - (1 - e^-(b-a)) * y)
    #            = a - log(1 - (1 - e^-(b-a)) * y)
    #            = a - log(1 + expm1(-(b-a)) * y)
    unif = np.random.random(size)
    if (b == np.inf):
        return a - np.log1p(-unif)
    else:
        return a - np.log1p(np.expm1(-(b - a)) * unif)

def sfs(demo, int S, int M, int n, float tau1, float tau2, extract_output=False, seed=None):
    logging.debug("in sfs")
    # Create stuff needed for computation
    # Sample conditionally; populate the interpolating rate matrices
    t1, t2 = [demo.eta.inverse_rate(x) for x in (tau1, tau2)]
    if seed:
        np.random.seed(seed)
    y = aca(exp1_conditional(t1, t2, M))
    mats, ts = moran_model.interpolators(n)
    ts = aca(ts)
    ei = aca(np.searchsorted(ts, y), dtype=np.int32)
    cdef double[::1] my = y
    cdef int[::1] mei = ei
    cdef double[::1] mts = ts
    cdef vector[double*] eM
    cdef double[:, :, ::1] mmats = aca(mats)
    cdef int i
    for i in range(mats.shape[0]):
        eM.push_back(&mmats[i, 0, 0])
    ts = aca(ts)
    cdef PiecewiseExponentialWrapper eta = demo.eta
    cdef ConditionedSFS* csfs = new ConditionedSFS(eta.pexp, n)
    if seed:
        csfs.set_seed(seed)
    csfs.compute(S, M, &my[0], eM, &mei[0], &mts[0])
    cdef ConditionedSFSWrapper ret

    # If we are not extracting output (i.e., for testing purposes)
    # then return a (wrapped) pointer to the csfs object for use
    # later on in the forward algorithm.
    if not extract_output:
        ret = ConditionedSFSWrapper()
        ret.csfs = csfs
        return ret

    K = len(demo.sqrt_a)
    sfs = aca(np.zeros((3, n + 1)))
    jac = aca(np.zeros((3, n + 1, 3, K)))
    cdef double[:, ::1] msfs = sfs
    cdef double[:, :, :, ::1] mjac = jac
    csfs.store_results(&msfs[0, 0], &mjac[0, 0, 0, 0])
    del csfs
    reduced_sfs = np.zeros(n + 1)
    for i in range(3):
        for j in range(n + 1):
            if i + j < n + 1:
                reduced_sfs[i + j] += sfs[i][j]
    return (reduced_sfs, sfs, jac)

def transition(demo, hidden_states, double rho, extract_output=False):
    cdef PiecewiseExponentialWrapper eta = demo.eta
    cdef vector[double] hs = hidden_states
    assert hidden_states[0] == 0.0
    assert hidden_states[-1] == np.inf
    cdef Transition* trans = new Transition(eta.pexp, hs, rho)
    trans.compute()
    cdef TransitionWrapper ret
    if not extract_output:
        ret = TransitionWrapper()
        ret.trans = transition
        return ret
    M = len(hidden_states)
    tmat = aca(np.zeros((M - 1, M - 1)))
    tjac = aca(np.zeros((M - 1, M - 1, 3, demo.K)))
    cdef double[:, ::1] tmatv = tmat
    cdef double[:, :, :, ::1] tjacv = tjac
    trans.store_results(&tmatv[0, 0], &tjacv[0, 0, 0, 0])
    del trans
    return (tmat, tjac)

def hmm(demo, sfs_list, obs, hidden_states, rho, theta, viterbi=False):
    print("in hmm")
    obs = aca(obs, dtype=np.int32)
    L = obs.shape[0]
    assert obs.shape == (L, 2)
    cdef int[:, ::1] mobs = obs
    cdef PiecewiseExponentialWrapper eta = demo.eta
    logging.debug("constructing hmm")
    cdef vector[vector[ConditionedSFSptr]] csfs
    cdef ConditionedSFSWrapper cw
    for k, sfs in enumerate(sfs_list):
        csfs.emplace_back()
        for sample in sfs:
            cw = sample
            csfs[k].push_back(cw.csfs)
    cdef HMM *myHmm = new HMM(eta.pexp, csfs, hidden_states, L, &mobs[0, 0], rho, theta)
    logging.debug("running forward algorithm")
    jac = aca(np.zeros((3, demo.K)))
    cdef double[:, ::1] mjac = jac
    logp = myHmm.logp(&mjac[0, 0])
    del myHmm
    return (logp, jac)
