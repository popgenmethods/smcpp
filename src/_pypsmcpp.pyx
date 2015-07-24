import random
import sys
cimport numpy as np
import numpy as np
import logging
import moran_model
import collections
import scipy.optimize

init_eigen();

logger = logging.getLogger(__name__)

# Everything needs to be C-order contiguous to pass in as
# flat arrays
aca = np.ascontiguousarray

cdef vector[double*] make_mats(mats):
    cdef vector[double*] expM
    cdef double[:, :, ::1] mmats = aca(mats)
    cdef int i
    for i in range(mats.shape[0]):
        expM.push_back(&mmats[i, 0, 0])
    return expM

cdef ParameterVector make_params(params):
    cdef vector[vector[double]] ret
    for p in params:
        ret.push_back(p)
    return ret

cdef _make_em_matrix(vector[pMatrixD] mats):
    cdef double[:, ::1] v
    ret = []
    for i in range(mats.size()):
        m = mats[i][0].rows()
        n = mats[i][0].cols()
        ary = aca(np.zeros([m, n]))
        v = ary
        store_matrix(mats[i], &v[0, 0])
        ret.append(ary)
    return ret

cdef class PyInferenceManager:
    cdef InferenceManager *_im
    cdef int _n, _nder
    cdef int _num_hmms
    cdef object _moran_mats
    cdef object _moran_ts
    cdef object _observations
    cdef np.ndarray _emission_mask
    cdef public long long seed

    def __cinit__(self, int n, observations, hidden_states, 
            double theta, double rho, int block_size, int num_threads, int num_samples,
            int mask_freq, mask_offset, emission_mask = None):
        self.seed = 1
        self._n = n
        cdef int[:, ::1] vob
        cdef vector[int*] obs
        cdef int L = observations[0].shape[0]
        self._observations = observations
        for ob in observations:
            if np.isfortran(ob):
                raise ValueError("Input arrays must be C-ordered")
            if ob.shape[0] != L:
                raise ValueError("Input data sets should all have the same shape")
            vob = ob
            obs.push_back(&vob[0, 0])
        self._num_hmms = len(observations)
        mats, ts = moran_model.interpolators(n)
        self._moran_mats = mats
        self._moran_ts = ts
        if emission_mask is None:
            emission_mask = np.arange(3 * (n + 1)).reshape([3, n + 1])
        self._emission_mask = aca(np.array(emission_mask, dtype=np.int32))
        cdef int[:, ::1] emv = self._emission_mask
        self._im = new InferenceManager(
                MatrixInterpolator(n + 1, self._moran_ts, make_mats(self._moran_mats)), 
                n, L, obs, hidden_states, &emv[0, 0], mask_freq, mask_offset, theta, rho, block_size, 
                num_threads, num_samples)

    def sfs(self, params, double t1, double t2, jacobian=False):
        set_csfs_seed(self.seed)
        cdef ParameterVector p = make_params(params)
        cdef Matrix[double] sfs
        cdef Matrix[adouble] dsfs
        ret = aca(np.zeros([3, self._n + 1]))
        cdef double[:, ::1] vret = ret
        if not jacobian:
            sfs = self._im.sfs_cython(p, t1, t2)
            store_matrix(&sfs, &vret[0, 0])
            return ret
        J = len(params) * len(params[0])
        jac = aca(np.zeros([3, self._n + 1, J]))
        cdef double[:, :, ::1] vjac = jac
        dsfs = self._im.dsfs_cython(p, t1, t2)
        store_sfs_results(dsfs, &vret[0, 0], &vjac[0, 0, 0])
        return ret, jac

    def getObservations(self):
        return self._observations

    def setParams(self, params, derivatives):
        # if not np.all(np.array(params) > 0):
            # raise ValueError("All parameters must be strictly positive")
        cdef ParameterVector p = make_params(params)
        set_csfs_seed(self.seed)
        if derivatives is True:
            derivatives = [(a, b) for a in range(len(params)) for b in range(len(params[0]))]
        if derivatives:
            # It should be pairs of tuples in this case
            self._nder = len(derivatives)
            self._im.setParams_ad(p, derivatives)
        else:
            self._im.setParams_d(p)

    def set_num_samples(self, int nsamples):
        self._im.set_num_samples(nsamples)

    def setDebug(self, val):
        self._im.debug = val

    def Estep(self):
        self._im.Estep()

    def alphas(self):
        return _make_em_matrix(self._im.getAlphas())

    def betas(self):
        return _make_em_matrix(self._im.getBetas())

    def gammas(self):
        return _make_em_matrix(self._im.getGammas())

    # def Bs(self):
    #     cdef vector[pMatrixAd] mats = self._im.getBs()
    #     cdef double[:, ::1] v
    #     ret = []
    #     for i in range(mats.size()):
    #         m = mats[i][0].rows()
    #         n = mats[i][0].cols()
    #         ary = aca(np.zeros([m, n]))
    #         v = ary
    #         store_matrix(mats[i], &v[0, 0])
    #         ret.append(ary)
    #     return ret

    def pi(self):
        cdef Matrix[double] mat = self._im.getPi()
        cdef double[:, ::1] v
        m = mat.rows()
        n = mat.cols()
        ary = aca(np.zeros([m, n]))
        v = ary
        store_matrix(&mat, &v[0, 0])
        return ary

    def transition(self):
        cdef Matrix[double] mat = self._im.getTransition()
        cdef double[:, ::1] v
        m = mat.rows()
        n = mat.cols()
        ary = aca(np.zeros([m, n]))
        v = ary
        store_matrix(&mat, &v[0, 0])
        return ary

    def emission(self):
        cdef Matrix[double] mat = self._im.getEmission()
        cdef double[:, ::1] v
        m = mat.rows()
        n = mat.cols()
        ary = aca(np.zeros([m, n]))
        v = ary
        store_matrix(&mat, &v[0, 0])
        return ary

    def masked_emission(self):
        cdef Matrix[double] mat = self._im.getMaskedEmission()
        cdef double[:, ::1] v
        m = mat.rows()
        n = mat.cols()
        ary = aca(np.zeros([m, n]))
        v = ary
        store_matrix(&mat, &v[0, 0])
        return ary

    def _call_inference_func(self, func, lam):
        if func == "loglik":
            return self._im.loglik(lam)
        cdef vector[adouble] ad_rets = self._im.Q(lam)
        cdef int K = ad_rets.size()
        ret = []
        cdef double[::1] vjac
        for i in range(self._num_hmms):
            jac = aca(np.zeros([self._nder]))
            vjac = jac
            fill_jacobian(ad_rets[i], &vjac[0])
            ret.append((toDouble(ad_rets[i]), jac))
        return ret

    def Q(self, lam):
        return self._call_inference_func("Q", lam)

    def loglik(self, lam):
        return self._call_inference_func("loglik", lam)

    def __dealloc__(self):
        del self._im

    def balance_hidden_states(self, params, int M):
        cdef ParameterVector p = make_params(params)
        ret = [0.0]
        t = 0
        T_MAX = 100
        for m in range(1, M):
            def f(t):
                return np.exp(-self._im.R(params, t)) - 1.0 * (M - m) / M
            res = scipy.optimize.brentq(f, ret[-1], T_MAX)
            ret.append(res)
        ret.append(np.inf)
        return np.array(ret)

def reduced_sfs(sfs):
    n = sfs.shape[1] - 1
    new_shape = [n + 2] + list(sfs.shape[2:])
    reduced_sfs = np.zeros(new_shape)
    for i in range(3):
        for j in range(n + 1):
            if 0 <= i + j < n + 2:
                reduced_sfs[i + j] += sfs[i][j]
    return reduced_sfs

def sfs(params, int n, int num_samples, double tau1, double tau2, int numthreads, double theta, seed=None, jacobian=False):
    K = len(params[0])
    for p in params:
        assert len(p) == K
    cdef vector[vector[double]] cparams = make_params(params)
    mats, ts = moran_model.interpolators(n)
    if not seed:
        seed = np.random.randint(0, sys.maxint)
    set_seed(seed)
    sfs = aca(np.zeros([3, n + 1]))
    cdef double[:, ::1] msfs = sfs
    cdef double[:, :, :, ::1] mjac 
    if jacobian:
        jac = aca(np.zeros((3, n + 1, len(params), K)))
        mjac = jac
        cython_calculate_sfs_jac(cparams, n, num_samples, MatrixInterpolator(n + 1, ts, make_mats(mats)),
                tau1, tau2, numthreads, theta, &msfs[0, 0], &mjac[0, 0, 0, 0])
        return (sfs, reduced_sfs(sfs), jac, reduced_sfs(jac))
    else:
        cython_calculate_sfs(cparams, n, num_samples, MatrixInterpolator(n + 1, ts, make_mats(mats)), 
                tau1, tau2, numthreads, theta, &msfs[0, 0])
        return (sfs, reduced_sfs(sfs))

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

def set_csfs_seed(long long seed):
    set_seed(seed)
