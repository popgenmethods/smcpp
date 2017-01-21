from __future__ import absolute_import, division, print_function
cimport numpy as np
from cython.operator cimport dereference as deref, preincrement as inc
import random
import sys
import numpy as np
import collections
import scipy.optimize
import collections
import wrapt
import six
import os.path
from appdirs import AppDirs
from ad import adnumber, ADF

from . import logging, version
from .observe import targets

logger = logging.getLogger(__name__)
logger.debug("SMC++ " + version.version)

init_eigen()

def _init_cache():
    dirs = AppDirs("smcpp", "popgenmethods", version=version.version)
    try:
        os.makedirs(dirs.user_cache_dir)
    except OSError:
        pass
    init_cache(os.path.join(dirs.user_cache_dir, "matrices.dat").encode("UTF-8"))
_init_cache()

abort = False
_lvl = {s: getattr(logging, s) for s in "info debug critical warning error".upper().split()}
_lvl['DEBUG'] = logging.DEBUG - 1
_lvl['DEBUG1'] = logging.DEBUG
cdef void logger_cb(const char* name, const char* level, const char* message) with gil:
    global abort
    name_s = "smcpp._smcpp:" + name.decode("UTF-8")
    level_s = level.decode("UTF-8")
    message_s = message.decode("UTF-8")
    try:
        logging.getLogger(name_s).log(_lvl[level_s.upper()], message_s)
    except KeyboardInterrupt:
        logging.getLogger(name_s).critical("Aborting")
        abort = True

def _check_abort():
    global abort
    try:
        if abort:
            raise KeyboardInterrupt()
    finally:
        abort = False

init_logger_cb(logger_cb);

# Everything needs to be C-order contiguous to pass in as
# flat arrays
aca = np.ascontiguousarray

cdef ParameterVector make_params(a, s, dlist) except *:
    cdef ParameterVector ret
    cdef vector[adouble] r
    assert len(a) > 0
    for aa in a:
        if not isinstance(aa, ADF):
            aa = adnumber(aa)
        r.push_back(double_vec_to_adouble(aa.x, [aa.d(da) for da in dlist]))
    ret.push_back(r)
    cdef vector[adouble] cs
    for ss in s:
        cs.push_back(adouble(ss))
    ret.push_back(cs)
    return ret

cdef ParameterVector make_params_from_model(model) except *:
    return make_params(model.stepwise_values(), model.s, model.dlist)

cdef _make_em_matrix(vector[pMatrixD] mats):
    cdef double[:, ::1] v
    ret = []
    for i in range(mats.size()):
        m = mats[i][0].rows()
        n = mats[i][0].cols()
        ary = aca(np.zeros([m, n]))
        v = ary
        store_matrix(mats[i][0], &v[0, 0])
        ret.append(ary)
    return ret

cdef object _adouble_to_ad(const adouble x, dlist):
    if len(dlist) == 0:
        return x.value()
    r = adnumber(x.value())
    cdef double[::1] vjac
    jac = aca(np.zeros([len(dlist)]))
    vjac = jac
    fill_jacobian(x, &vjac[0])
    for i, d in enumerate(dlist):
        r.d()[d] = jac[i]
    return r

cdef _store_admatrix_helper(Matrix[adouble] &mat, dlist):
    nder = len(dlist)
    m = mat.rows()
    n = mat.cols()
    cdef int i
    cdef int j
    ary = aca(np.zeros([m, n], dtype=object))
    for i in range(m):
        for j in range(n):
            ary[i, j] = _adouble_to_ad(mat(i, j), dlist)
    return ary


cdef class _PyInferenceManager:
    cdef int _num_hmms
    cdef object _model, _observations, _theta, _rho, _im_id
    cdef public long long seed
    cdef vector[double] _hs
    cdef vector[int] _Ls
    cdef InferenceManager* _im
    cdef vector[int*] _obs_ptrs

    def __my_cinit__(self, observations, hidden_states, im_id=None):
        self._im_id = im_id
        self.seed = 1
        cdef int[:, ::1] vob
        if len(observations) == 0:
            raise RuntimeError("Observations list is empty")
        self._observations = observations
        Ls = []
        ## Validate hidden states
        if any([not np.all(np.sort(hidden_states) == hidden_states),
            hidden_states[0] != 0., hidden_states[-1] != np.inf]):
            raise RuntimeError("Hidden states must be in ascending order with hs[0]=0 and hs[-1] = infinity")
        for ob in observations:
            vob = ob
            self._obs_ptrs.push_back(&vob[0, 0])
            Ls.append(ob.shape[0])
        self._num_hmms = len(observations)
        self._hs = hidden_states
        self._Ls = Ls
        _check_abort()

    def __dealloc__(self):
        del self._im

    property observations:
        def __get__(self):
            return self._observations

    property theta:
        def __get__(self):
            return self._theta

        def __set__(self, theta):
            self._theta = theta
            self._im.setTheta(theta)

    property rho:
        def __get__(self):
            return self._rho

        def __set__(self, rho):
            self._rho = rho
            self._im.setRho(rho)

    def E_step(self, forward_backward_only=False):
        cdef bool fbOnly = forward_backward_only
        with nogil:
            self._im.Estep(fbOnly)
        _check_abort()

    property model:
        def __get__(self):
            return self._model
        def __set__(self, m):
            self._model = m
            m.register(self)

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
            ret = {}
            while it != ep.end():
                bk = []
                for i in range(deref(it).first.size()):
                    bk.append(deref(it).first(i))
                M = deref(it).second.size()
                v = np.zeros(M, dtype=object)
                for i in range(M):
                    v[i] = _adouble_to_ad(deref(it).second(i), self._model.dlist)
                ret[tuple(bk)] = v
                inc(it)
            return ret


    property gamma_sums:
        def __get__(self):
            ret = []
            cdef vector[pBlockMap] gs = self._im.getGammaSums()
            cdef vector[pBlockMap].iterator it = gs.begin()
            cdef map[block_key, Vector[double]].iterator map_it
            cdef double[::1] vary
            cdef int M = len(self.hidden_states) - 1
            while it != gs.end():
                map_it = deref(it).begin()
                pairs = {}
                while map_it != deref(it).end():
                    bk = []
                    ary = np.zeros(M)
                    for i in range(deref(map_it).first.size()):
                        bk.append(deref(map_it).first(i))
                    for i in range(M):
                        ary[i] = deref(map_it).second(i)
                    pairs[tuple(bk)] = ary
                    inc(map_it)
                inc(it)
                ret.append(pairs)
            return ret

    property gammas:
        def __get__(self):
            return _make_em_matrix(self._im.getGammas())

    def getXisums(self):
        return self.xisums

    property xisums:
        def __get__(self):
            return _make_em_matrix(self._im.getXisums())

    property pi:
        def __get__(self):
            return _store_admatrix_helper(self._im.getPi(), self._model.dlist)

    property transition:
        def __get__(self):
            return _store_admatrix_helper(self._im.getTransition(), self._model.dlist)

    property emission:
        def __get__(self):
            return _store_admatrix_helper(self._im.getEmission(), self._model.dlist)

    def Q(self, separate=False):
        cdef vector[adouble] ad_rets
        try:
            with nogil:
                ad_rets = self._im.Q()
        except RuntimeError as e:
            if str(e) == "SFS is not a probability distribution":
                logger.debug("Model does not induce a valid probability distribution")
                return adnumber(-np.inf)
            raise
        _check_abort()
        cdef int i
        cdef adouble q = adouble(0)
        qq = []
        for i in range(3):
            z = _adouble_to_ad(ad_rets[i], self._model.dlist)
            qq.append(z)
            q += ad_rets[i]
            logger.debug(("im(%r).q%d" % (self._im_id, i + 1), z, [z.d(x) for x in self._model.dlist]))
        if separate:
            return qq
        r = adnumber(toDouble(q))
        if self._model.dlist:
            r = _adouble_to_ad(q, self._model.dlist)
        return r

    def loglik(self):
        cdef vector[double] llret
        with nogil:
            llret = self._im.loglik()
        _check_abort()
        return sum(llret)

cdef class PyOnePopInferenceManager(_PyInferenceManager):

    def __cinit__(self, int n, observations, hidden_states, im_id, bool fold):
        # This is needed because cinit cannot be inherited
        self.__my_cinit__(observations, hidden_states, im_id)
        with nogil:
            self._im = new OnePopInferenceManager(n, self._Ls, self._obs_ptrs, self._hs, fold)

    @property
    def pid(self):
        assert len(self._im_id[0]) == 1
        return self._im_id[0][0]

    @targets("model update")
    def update(self, message, *args, **kwargs):
        m = self._model.for_pop(self.pid)
        cdef ParameterVector params = make_params_from_model(m)
        with nogil:
            self._im.setParams(params)

cdef class PyTwoPopInferenceManager(_PyInferenceManager):

    cdef TwoPopInferenceManager* _im2
    cdef int _a1

    def __cinit__(self, int n1, int n2, int a1, int a2, observations, hidden_states, im_id, bool fold):
        # This is needed because cinit cannot be inherited
        assert a1 + a2 == 2
        assert a1 in [1, 2]
        assert a2 in [0, 1]
        self._a1 = a1
        self.__my_cinit__(observations, hidden_states, im_id)
        assert a1 in [1, 2], "a2=2 is not supported"
        with nogil:
            self._im2 = new TwoPopInferenceManager(n1, n2, a1, a2, self._Ls, self._obs_ptrs, self._hs, fold)
            self._im = self._im2

    @targets("model update")
    def update(self, message, *args, **kwargs):
        m = self._model
        pids = self._im_id[0]
        if self._a1 == 1:
            dist = None
        else:
            assert self._a1 == 2
            dist = pids[0]
        dm = m.for_pop(dist)
        cdef ParameterVector distinguished_params = make_params(dm.stepwise_values(), dm.s, m.dlist)
        ms = [m.for_pop(p) for p in pids]
        cdef ParameterVector params1 = make_params(ms[0].stepwise_values(), ms[0].s, m.dlist)
        cdef ParameterVector params2 = make_params(ms[1].stepwise_values(), ms[1].s, m.dlist)
        cdef double split = m.split
        with nogil:
            self._im2.setParams(distinguished_params, params1, params2, split)

cdef class PyRateFunction:
    cdef unique_ptr[PiecewiseConstantRateFunction[adouble]] _eta
    cdef object _model
    def __cinit__(self, model, hs):
        self._model = model
        cdef ParameterVector params = make_params_from_model(model)
        cdef vector[double] _hs = hs
        with nogil:
            self._eta.reset(new PiecewiseConstantRateFunction[adouble](params, _hs))

    def R(self, t):
        assert np.isfinite(t)
        cdef adouble Rt = self._eta.get().R(adouble(t))
        return _adouble_to_ad(Rt, self._model.dlist)

    def average_coal_times(self):
        cdef vector[adouble] v = self._eta.get().average_coal_times()
        return [toDouble(vv) for vv in v]

    def random_coal_times(self, t1, t2, K):
        cdef adouble t
        times = []
        for _ in range(K):
            ary = []
            t = self._eta.get().random_time(t1, t2, np.random.randint(six.MAXSIZE))
            ary = [_adouble_to_ad(t, self._model.dlist)]
            t = self._eta.get().R(t)
            ary.append(_adouble_to_ad(t, self._model.dlist))
            times.append(ary)
        return times

def raw_sfs(model, int n, double t1, double t2, below_only=False):
    cdef ParameterVector pv = make_params_from_model(model)
    cdef Matrix[adouble] dsfs
    cdef Matrix[double] sfs
    ret = aca(np.zeros([3, n + 1]))
    cdef double[:, ::1] vret = ret
    cdef vector[pair[int, int]] derivs
    cdef bool bo = below_only
    with nogil:
        dsfs = sfs_cython(n, pv, t1, t2, bo)
    _check_abort()
    return _store_admatrix_helper(dsfs, model.dlist)


def thin_data(data, int thinning, int offset=0):
    '''Implement the thinning procedure needed to break up correlation
    among the full SFS emissions.'''
    # Thinning
    cdef int i = offset
    out = []
    cdef int[:, :] vdata = data
    cdef int j
    cdef int k = data.shape[0]
    cdef int span
    cdef int npop = (data.shape[1] - 1) / 3
    cdef int sa
    a = np.zeros(npop, dtype=np.int32)
    b = np.zeros(npop, dtype=np.int32)
    nb = np.zeros(npop, dtype=np.int32)
    thin = np.zeros(npop * 3, dtype=np.int32)
    nonseg = np.zeros(npop * 3, dtype=np.int32)
    cdef int[:] a_view = a, b_view = b, nb_view = nb, thin_view = thin, nonseg_view = nonseg
    for j in range(k):
        span = vdata[j, 0]
        a_view[:] = vdata[j, 1::3]
        b_view[:] = vdata[j, 2::3]
        nb_view[:] = vdata[j, 3::3]
        sa = a.sum()
        thin_view[::3] = a_view
        if sa == 2:
            thin_view[::3] = 0
        while span > 0:
            if i < thinning and i + span >= thinning:
                if thinning - i > 1:
                    out.append([thinning - i - 1] + list(thin_view))
                if sa == 2 and np.all(b == nb):
                    nonseg_view[2::3] = nb_view
                    out.append([1] + list(nonseg_view))
                else:
                    out.append([1] + list(vdata[j, 1:]))
                span -= thinning - i
                i = 0
            else:
                out.append([span] + list(thin_view))
                i += span
                break
    ret = np.array(out, dtype=np.int32)
    assert ret[:, 0].sum() == data[:, 0].sum()
    return ret

# Used for testing purposes only
def joint_csfs(int n1, int n2, int a1, int a2, model, hidden_states, int K=10):
    assert (a1 == 2 and a2 == 0) or (a1 == a2 == 1)
    cdef vector[double] hs = hidden_states
    cdef ParameterVector p1 = make_params_from_model(model.model1)
    cdef ParameterVector p2 = make_params_from_model(model.model2)
    cdef double split = model.split
    cdef vector[Matrix[adouble]] jc
    cdef PiecewiseConstantRateFunction[adouble] *eta
    cdef JointCSFS[adouble] *jcsfs
    with nogil:
        eta = new PiecewiseConstantRateFunction[adouble](p1, hs)
        jcsfs = new JointCSFS[adouble](n1, n2, a1, a2, hs, K)
        jcsfs.pre_compute(p1, p2, split)
        jc = jcsfs.compute(deref(eta))
        del eta
        del jcsfs
    ret = []
    for i in range(jc.size()):
        mat = _store_admatrix_helper(jc[i], model.dlist)
        mat.shape = (a1 + 1, n1 + 1, a2 + 1, n2 + 1)
        ret.append(mat)
    return ret
