cimport numpy as np
from cython.operator cimport dereference as deref, preincrement as inc

import random
import sys
import numpy as np
import logging
import collections
import scipy.optimize
import collections
import wrapt
from ad import adnumber

T_MAX = C_T_MAX - 0.1

init_eigen();
logger = logging.getLogger(__name__)

abort = False
cdef void logger_cb(const char* name, const char* level, const char* message) with gil:
    global abort
    try:
        lvl = {"INFO": logging.INFO, "DEBUG": logging.DEBUG - 1, "WARNING": logging.WARNING}
        logging.getLogger(name).log(lvl[level.upper()], message)
    except KeyboardInterrupt:
        logging.getLogger(name).critical("Aborting")
        abort = True

init_logger_cb(logger_cb);

# Everything needs to be C-order contiguous to pass in as
# flat arrays
aca = np.ascontiguousarray

cdef struct ParameterBundle:
    vector[vector[double]] vals
    vector[pair[int, int]] derivs

cdef ParameterBundle make_params(model):
    cdef ParameterBundle ret
    for i in range(3):
        r = []
        for j in range(model.K):
            entry = model[i, j]
            r.append(float(entry))
        ret.vals.push_back(r)
    for b in range(model.K):
        ret.derivs.push_back((0, b))
        if b not in model.flat_pieces:
            ret.derivs.push_back((1, b))
    return ret

cdef _make_em_matrix(vector[pMatrixD] mats):
    cdef double[:, ::1] v
    ret = []
    for i in range(mats.size()):
        m = mats[i][0].rows()
        n = mats[i][0].cols()
        ary = aca(np.zeros([m, n]))
        v = ary
        store_matrix[double](mats[i], &v[0, 0])
        ret.append(ary)
    return ret

def validate_observation(ob):
    if np.isfortran(ob):
        raise ValueError("Input arrays must be C-ordered")
    if np.any(np.logical_and(ob[:, 1] == 2, ob[:, 2] == ob[:, 3])):
        raise RuntimeError("Error: data set contains sites where every individual is homozygous recessive. "
                           "Please encode / fold these as non-segregating (homozygous dominant).")

@wrapt.decorator
def modelify(wrapped, instance, args, kwargs):
    model = args[0]
    if isinstance(model, (np.ndarray, collections.Sequence)):
        from .model import SMCModel
        m = SMCModel(model[2], np.where(model[0] != model[1])[0])
        m.x[:2] = model[:2]
        model = m
    new_args = (model,) + args[1:]
    return wrapped(*new_args, **kwargs)

cdef class PyInferenceManager:
    cdef InferenceManager *_im
    cdef int _n
    cdef int _num_hmms
    cdef object _observations, _derivatives, _model
    cdef public long long seed

    def __cinit__(self, int n, observations, hidden_states, double theta, double rho):
        self.seed = 1
        self._n = n
        cdef int[:, ::1] vob
        cdef vector[int*] obs
        if len(observations) == 0:
            raise RuntimeError("Observations list is empty")
        self._observations = observations
        Ls = []
        ## Validate hidden states
        if any([not np.all(np.sort(hidden_states) == hidden_states),
            hidden_states[0] != 0., hidden_states[-1] > T_MAX]):
            raise RuntimeError("Hidden states must be in ascending order with hs[0]=0 and hs[-1] < %g" % T_MAX)
        for ob in observations:
            validate_observation(ob)
            vob = ob
            obs.push_back(&vob[0, 0])
            Ls.append(ob.shape[0])
        self._num_hmms = len(observations)
        cdef vector[double] hs = hidden_states
        cdef vector[int] _Ls = Ls
        with nogil:
            self._im = new InferenceManager(n, _Ls, obs, hs, theta, rho)

    def __dealloc__(self):
        del self._im

    def get_observations(self):
        return self._observations

    @modelify
    def set_params(self, model, theta, rho, dmodel, dtheta, drho, skip_emission=False):
        global abort
        if abort:
            abort = False
            raise KeyboardInterrupt
        if not np.all(np.array(model.x) > 0):
            raise ValueError("All parameters must be strictly positive")
        cdef ParameterBundle pb = make_params(model)
        self._model = model
        self._derivatives = []
        if dmodel:
            self._derivatives += list(pb.derivs)
        if drho:
            self._derivatives += [(3,0)]
        if dtheta:
            self._derivatives += [(4,0)]
        vector[pair[int, int]] d = self._derivatives
        if self._derivatives:
            with nogil:
                self._im.setParams_ad(pb.vals, theta, rho, d, skip_emission)
        else:
            with nogil:
                self._im.setParams_d(pb.vals, theta, rho, skip_emission)

    def E_step(self, forward_backward_only=False):
        logger.debug("Forward-backward algorithm...")
        cdef bool fbOnly = forward_backward_only
        with nogil:
            self._im.Estep(fbOnly)
        logger.debug("Forward-backward algorithm finished.")

    property span_cutoff:
        def __get__(self):
            return self._im.spanCutoff
        def __set__(self, bint sc):
            self._im.spanCutoff = sc

    property save_gamma:
        def __get__(self):
            return self._im.saveGamma
        def __set__(self, bint sg):
            self._im.saveGamma = sg

    property hidden_states:
        def __get__(self):
            return self._im.hidden_states
        def __set__(self, hs):
            self._im.hidden_states = hs

    property emission_probs:
        def __get__(self):
            cdef map[block_key, Vector[adouble]] ep = self._im.getEmissionProbs()
            cdef map[block_key, Vector[adouble]].iterator it = ep.begin()
            cdef pair[block_key, Vector[adouble]] p
            ret = {}
            while it != ep.end():
                p = deref(it)
                key = [0] * 3
                for i in range(3):
                    key[i] = p.first[i]
                M = p.second.size()
                v = np.zeros(M)
                if self._derivatives:
                    dv = np.zeros([M, self._nder])
                for i in range(M):
                    v[i] = p.second(i).value()
                    for j in range(self._nder):
                        dv[i, j] = p.second(i).derivatives()(j)
                ret[tuple(key)] = (v, dv) if self._nder else v
                inc(it)
            return ret


    property gamma_sums:
        def __get__(self):
            ret = []
            cdef vector[pBlockMap] gs = self._im.getGammaSums()
            cdef vector[pBlockMap].iterator it = gs.begin()
            cdef map[block_key, Vector[double]].iterator map_it
            cdef pair[block_key, Vector[double]] p
            cdef double[::1] vary
            cdef int M = len(self.hidden_states) - 1
            while it != gs.end():
                map_it = deref(it).begin()
                pairs = {}
                while map_it != deref(it).end():
                    p = deref(map_it)
                    bk = [0, 0, 0]
                    ary = np.zeros(M)
                    for i in range(3):
                        bk[i] = p.first[i]
                    for i in range(M):
                        ary[i] = p.second(i)
                    pairs[tuple(bk)] = ary
                    inc(map_it)
                inc(it)
                ret.append(pairs)
            return ret

    property gammas:
        def __get__(self):
            return _make_em_matrix(self._im.getGammas())

    property xisums:
        def __get__(self):
            return _make_em_matrix(self._im.getXisums())

    property pi:
        def __get__(self):
            return _store_admatrix_helper(self._im.getPi(), self._nder)

    property transition:
        def __get__(self):
            return _store_admatrix_helper(self._im.getTransition(), self._nder)

    property emission:
        def __get__(self):
            return _store_admatrix_helper(self._im.getEmission(), self._nder)

    def _call_inference_func(self, func):
        cdef vector[double] llret
        if func == "loglik":
            with nogil:
                llret = self._im.loglik()
            return llret
        cdef vector[adouble] ad_rets 
        with nogil:
            ad_rets = self._im.Q()
        cdef int K = ad_rets.size()
        ret = []
        cdef double[::1] vjac
        for i in range(self._num_hmms):
            r = toDouble(ad_rets[i])
            if self._derivatives:
                r = adnumber(r)
                jac = aca(np.zeros([len(self._derivatives)]))
                vjac = jac
                fill_jacobian(ad_rets[i], &vjac[0])
                d = {}
                for k, cc in enumerate(self._derivatives):
                    for var in self._model[cc].d():
                        d[var] = d.get(var, 0) + self._model[cc].d(var) * jac[k]
                r.d().update(d)
            ret.append(r)
        return ret

    def Q(self):
        return self._call_inference_func("Q")

    def loglik(self):
        return self._call_inference_func("loglik")

@modelify
def balance_hidden_states(model, int M):
    M -= 1
    cdef ParameterBundle pb = make_params(model)
    cdef vector[double] v = []
    cdef PiecewiseExponentialRateFunction[double] *eta = new PiecewiseExponentialRateFunction[double](pb.vals, v)
    try:
        ret = [0.0]
        t = 0
        for m in range(1, M):
            def f(double t):
                cdef double Rt = eta.R(t)
                return np.exp(-Rt) - 1.0 * (M - m) / M
            res = scipy.optimize.brentq(f, ret[-1], T_MAX)
            ret.append(res)
    finally:
        del eta
    if ret[-1] < T_MAX:
        ret.append(T_MAX)
    return np.array(ret)

@modelify
def sfs(model, int n, double t1, double t2, double theta, jacobian=False):
    cdef ParameterBundle pb = make_params(model)
    cdef Matrix[double] sfs
    cdef Matrix[adouble] dsfs
    ret = aca(np.zeros([3, n - 1]))
    cdef double[:, ::1] vret = ret
    if not jacobian:
        with nogil:
            sfs = sfs_cython[double](n, pb.vals, t1, t2, theta)
        store_matrix(&sfs, &vret[0, 0])
        return ret
    with nogil:
        dsfs = sfs_cython[adouble](n, pb.vals, t1, t2, theta, pb.derivs)
    J = pb.derivs.size()
    ret, jac = _store_admatrix_helper(dsfs, J)
    ret = adnumber(ret)
    derivs = list(pb.derivs)
    for i in range(3):
        for j in range(n - 1):
            for k, p in enumerate(derivs):
                for v in model[p].d():
                    ret[i, j].d()[v] = ret[i, j].d().get(v, 0.) + model[p].d(v) * jac[i, j, k]
    return ret

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

def thin_data(data, int thinning, int offset=0):
    '''Implement the thinning procedure needed to break up correlation
    among the full SFS emissions.'''
    # Thinning
    cdef int i = offset
    out = []
    cdef int[:, :] vdata = data
    cdef int k = data.shape[0]
    cdef int span, a, b, nb, a1
    for j in range(k):
        span = vdata[j, 0]
        a = vdata[j, 1]
        b = vdata[j, 2]
        nb = vdata[j, 3]
        a1 = a
        if a1 == 2:
            a1 = 0
        while span > 0:
            if i < thinning and i + span >= thinning:
                if thinning - i > 1:
                    out.append([thinning - i - 1, a1, 0, 0])
                if a == 2 and b == nb:
                    out.append([1, 0, 0, nb])
                else:
                    out.append([1, a, b, nb])
                span -= thinning - i
                i = 0
            else:
                out.append([span, a1, 0, 0])
                i += span
                break
    return np.array(out, dtype=np.int32)
