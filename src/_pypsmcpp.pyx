import random
import sys
cimport numpy as np
import numpy as np
import logging
import moran_model
import collections

logger = logging.getLogger(__name__)

# Everything needs to be C-order contiguous to pass in as
# flat arrays
aca = np.ascontiguousarray

cdef vector[vector[double]] make_params(params):
    cdef vector[vector[double]] ret
    for p in params:
        ret.push_back(p)
    return ret

cdef vector[double] from_list(lst):
    cdef vector[double] ret
    for l in lst:
        ret.push_back(l)
    return ret

def log_likelihood(params, int n, int S, int M, obs_list, hidden_states, 
        double rho, double theta, double reg_lambda,
        int numthreads, seed=None, viterbi=False, jacobian=False):
    # Create stuff needed for computation
    # Sample conditionally; populate the interpolating rate matrices
    J = len(params)
    K = len(params[0])
    for p in params:
        assert len(p) == K
    cdef vector[vector[double]] cparams = make_params(params)
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
    jac = aca(np.zeros((J, K)))
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
        ad = loglik[adouble](cparams,
                n, 
                S, M, 
                from_list(ts), expM, 
                L, vobs, 
                from_list(hidden_states), rho, theta, 
                numthreads,
                viterbi, viterbi_paths, 
                reg_lambda)
        fill_jacobian(ad, &mjac[0, 0])
        ret = (toDouble(ad), jac)
        if viterbi:
            ret += (np.sum(viterbi_paths, axis=0),)
    else:
        ret = loglik[double](
            cparams,
            n, 
            S, M, 
            from_list(ts), expM, 
            L, vobs, 
            from_list(hidden_states), rho, theta, 
            numthreads,
            viterbi, viterbi_paths, 
            reg_lambda)
    return ret

def _reduced_sfs(sfs):
    n = sfs.shape[1] - 1
    reduced_sfs = np.zeros(n + 1)
    for i in range(3):
        for j in range(n + 1):
            if i + j < n + 1:
                reduced_sfs[i + j] += sfs[i][j]
    return reduced_sfs

def sfs(params, int n, int S, int M, double tau1, double tau2, int numthreads, 
        double theta, seed=None, jacobian=False):
    K = len(params[0])
    for p in params:
        assert len(p) == K
    cdef vector[vector[double]] cparams = make_params(params)
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
        jac = aca(np.zeros((3, n + 1, len(params), K)))
        mjac = jac
        cython_calculate_sfs_jac(cparams, n, S, M, ts, expM, tau1, tau2, 
                numthreads, theta, &msfs[0, 0], &mjac[0, 0, 0, 0])
        return (sfs, _reduced_sfs(sfs), jac)
    else:
        cython_calculate_sfs(cparams, n, S, M, ts, expM, tau1, tau2, 
                numthreads, theta, &msfs[0, 0])
        return (sfs, _reduced_sfs(sfs))

def transition(params, hidden_states, rho, jacobian=False):
    J = len(params)
    K = len(params[0])
    for p in params:
        assert len(p) == K
    cdef vector[vector[double]] cparams = make_params(params)
    assert hidden_states[0] == 0.0
    assert hidden_states[-1] == np.inf
    M = len(hidden_states) - 1
    trans = aca(np.zeros([M, M]))
    cdef double[:, ::1] mtrans = trans
    cdef double[:, :, :, ::1] mjac
    if jacobian:
        jac = aca(np.zeros((M, M, J, K)))
        mjac = jac
        cython_calculate_transition_jac(cparams, hidden_states, rho, 
                &mtrans[0, 0], &mjac[0, 0, 0, 0])
        return (trans, jac)
    else:
        cython_calculate_transition(cparams, hidden_states, rho, &mtrans[0, 0])
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
