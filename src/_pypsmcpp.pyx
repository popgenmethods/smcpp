import random
import sys
cimport numpy as np
import numpy as np
import logging
import collections
import scipy.optimize

init_eigen();

logger = logging.getLogger(__name__)

# Everything needs to be C-order contiguous to pass in as
# flat arrays
aca = np.ascontiguousarray

def do_progress(x):
    doProgress(x)

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

cdef _make_em_matrix(vector[pMatrixF] mats):
    cdef fbType[:, ::1] v
    ret = []
    for i in range(mats.size()):
        m = mats[i][0].rows()
        n = mats[i][0].cols()
        ary = aca(np.zeros([m, n]))
        v = ary
        store_matrix[fbType](mats[i], &v[0, 0])
        ret.append(ary)
    return ret

def validate_observation(ob):
    if np.isfortran(ob):
        raise ValueError("Input arrays must be C-ordered")
    if np.any(np.logical_and(ob[:, 1] == 2, ob[:, 2] == ob[:, 3])):
        raise RuntimeError("Error: data set contains sites where every individual is homozygous recessive. "
                           "Please encode / fold these as non-segregating (homozygous dominant).")

cdef class PyInferenceManager:
    cdef InferenceManager *_im
    cdef int _n, _nder
    cdef int _num_hmms
    cdef object _observations
    cdef np.ndarray _emission_mask
    cdef public long long seed

    def __cinit__(self, int n, observations, hidden_states, 
            double theta, double rho, int block_size,
            int mask_freq, emission_mask = None):
        self.seed = 1
        self._n = n
        cdef int[:, ::1] vob
        cdef vector[int*] obs
        self._observations = observations
        Ls = []
        for ob in observations:
            validate_observation(ob)
            vob = ob
            obs.push_back(&vob[0, 0])
            Ls.append(ob.shape[0])
        self._num_hmms = len(observations)
        if emission_mask is None:
            emission_mask = np.arange(3 * (n + 1)).reshape([3, n + 1])
        self._emission_mask = aca(np.array(emission_mask, dtype=np.int32))
        cdef int[:, ::1] emv = self._emission_mask
        cdef vector[double] hs = hidden_states
        self._im = new InferenceManager(
                n, Ls, obs, hs, &emv[0, 0], mask_freq, theta, rho, block_size)


    def __dealloc__(self):
        del self._im

    def getObservations(self):
        return self._observations

    def setParams(self, params, derivatives):
        if not np.all(np.array(params) > 0):
            raise ValueError("All parameters must be strictly positive")
        if not all(len(pp) == len(params[0]) for pp in params):
            raise ValueError("All parameters must have same sizes")
        cdef ParameterVector p = make_params(params)
        if derivatives is True:
            derivatives = [(a, b) for a in range(len(params)) for b in range(len(params[0]))]
        if derivatives:
            # It should be pairs of tuples in this case
            self._nder = len(derivatives)
            self._im.setParams_ad(p, derivatives)
        else:
            self._nder = 0
            self._im.setParams_d(p)

    def setDebug(self, val):
        self._im.debug = val

    def regularizer(self):
        return self._im.getRegularizer()

    def Estep(self):
        self._im.Estep()

    def random_times(self, params, fac, size):
        cdef ParameterVector p = make_params(params)
        return self._im.randomCoalTimes(p, fac, size)

    property saveGamma:
        def __get__(self):
            return self._im.saveGamma
        def __set__(self, bint sg):
            self._im.saveGamma = sg

    property hidden_states:
        def __get__(self):
            return self._im.hidden_states
        def __set__(self, hs):
            self._im.hidden_states = hs

    property forwardOnly:
        def __get__(self):
            return self._im.forwardOnly
        def __set__(self, bint fo):
            self._im.forwardOnly = fo

    property gammas:
        def __get__(self):
            return _make_em_matrix(self._im.getGammas())

    property xisums:
        def __get__(self):
            cdef pair[vector[pMatrixF], vector[pMatrixF]] xis = self._im.getXisums()
            return [_make_em_matrix(xis.first), _make_em_matrix(xis.second)]

    property Bs:
        def __get__(self):
            cdef vector[pMatrixAd] mats = self._im.getBs()
            cdef double[:, ::1] v
            cdef double[:, :, ::1] av
            ret = []
            for i in range(mats.size()):
                ret.append(_store_admatrix_helper(mats[i][0], self._nder))
            return ret

    property block_keys:
        def __get__(self):
            return self._im.getBlockKeys()

    property pi:
        def __get__(self):
            return _store_admatrix_helper(self._im.getPi(), self._nder)

    property transition:
        def __get__(self):
            return _store_admatrix_helper(self._im.getTransition(), self._nder)

    property emission:
        def __get__(self):
            return _store_admatrix_helper(self._im.getEmission(), self._nder)

    def _call_inference_func(self, func, lam):
        if func == "loglik":
            return self._im.loglik(lam)
        cdef vector[adouble] ad_rets = self._im.Q(lam)
        cdef int K = ad_rets.size()
        ret = []
        cdef double[::1] vjac
        for i in range(self._num_hmms):
            if (self._nder > 0):
                jac = aca(np.zeros([self._nder]))
                vjac = jac
                fill_jacobian(ad_rets[i], &vjac[0])
                ret.append((toDouble(ad_rets[i]), jac))
            else:
                ret.append(toDouble(ad_rets[i]))
        return ret

    def Q(self, lam):
        return self._call_inference_func("Q", lam)

    def loglik(self, lam):
        return self._call_inference_func("loglik", lam)

    def balance_hidden_states(self, params, int M):
        cdef ParameterVector p = make_params(params)
        ret = [0.0]
        t = 0
        T_MAX = 14.9
        for m in range(1, M):
            def f(t):
                return np.exp(-self._im.R(params, t)) - 1.0 * (M - m) / M
            while np.sign(f(T_MAX)) == np.sign(f(0)):
                T_MAX *= 2
                print(T_MAX)
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

def sfs(int n, params, double t1, double t2, double theta, jacobian=False):
    cdef ParameterVector p = make_params(params)
    cdef Matrix[double] sfs
    cdef Matrix[adouble] dsfs
    ret = aca(np.zeros([3, n - 1]))
    cdef double[:, ::1] vret = ret
    if not jacobian:
        sfs = sfs_cython[double](n, p, t1, t2, theta)
        store_matrix(&sfs, &vret[0, 0])
        return ret
    J = len(jacobian)
    jac = aca(np.zeros([3, n - 1, J]))
    cdef double[:, :, ::1] vjac = jac
    dsfs = sfs_cython[adouble](n, p, t1, t2, theta, jacobian)
    return _store_admatrix_helper(dsfs, J)

cdef _store_admatrix_helper(Matrix[adouble] &mat, int nder):
    cdef double[:, ::1] v
    cdef double[:, :, ::1] av
    m = mat.rows()
    n = mat.cols()
    ary = aca(np.zeros([m, n]))
    v = ary
    if (nder == 0):
        store_admatrix(mat, nder, &v[0, 0], NULL)
        return ary
    else:
        jac = aca(np.zeros([m, n, nder]))
        av = jac
        store_admatrix(mat, nder, &v[0, 0], &av[0, 0, 0])
        return ary, jac
