from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.map cimport map
from libcpp cimport bool

cdef extern from "common.h":
    cdef cppclass Vector[T]:
        T& operator()(int)
        int size()
    adouble double_vec_to_adouble(double x, vector[double] v)
    cdef cppclass adouble:
        double value()
        Vector[double] derivatives()
    cdef cppclass Matrix[T]:
        int rows()
        int cols()
    cdef double toDouble(const adouble &)
    void init_eigen()
    void init_logger_cb(void(*)(const char*, const char*, const char*))
    void fill_jacobian(const adouble &, double*)
    void store_matrix(const Matrix[double]&, double*)
    void store_matrix(const Matrix[adouble]&, double*)
    void store_matrix(const Matrix[adouble]&, double*, double*)

ctypedef Matrix[double]* pMatrixD
ctypedef Matrix[adouble]* pMatrixAd
ctypedef map[block_key, Vector[double]]* pBlockMap

cdef extern from "inference_manager.h":
    cdef cppclass block_key:
        int& operator[](int)
    ctypedef vector[vector[adouble]] ParameterVector
    cdef cppclass InferenceManager nogil:
        InferenceManager(const int, const vector[int],
                const vector[int*], const vector[double],
                const vector[double]) except +
        void setParams(const ParameterVector)
        void setTheta(const double)
        void setRho(const double)
        void Estep(bool)
        vector[double] loglik()
        vector[adouble] Q()
        double R(const ParameterVector, double t)
        bool debug
        bool saveGamma
        bool folded
        int spanCutoff
        vector[double] hidden_states
        adouble getRegularizer()
        vector[pMatrixD] getGammas()
        vector[pMatrixD] getXisums()
        vector[pBlockMap] getGammaSums()
        Matrix[adouble]& getPi()
        Matrix[adouble]& getTransition()
        Matrix[adouble]& getEmission()
        map[block_key, Vector[adouble] ]& getEmissionProbs()
    Matrix[adouble] sfs_cython(int, const ParameterVector&, const vector[double], double, double) nogil


cdef extern from "piecewise_exponential_rate_function.h":
    cdef cppclass PiecewiseExponentialRateFunction[T]:
        PiecewiseExponentialRateFunction(const ParameterVector, const vector[double], const vector[double])
        T R(T x)
