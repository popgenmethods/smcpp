from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.map cimport map
from libcpp cimport bool

cdef extern from "common.h":
    cdef cppclass Vector[T]:
        T& operator()(int)
        int size()
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
    void store_matrix[T](const Matrix[T]*, T*)
    void store_admatrix(const Matrix[adouble]&, int, double*, double*)
    void doProgress(bool)
    const double T_MAX

ctypedef Matrix[double]* pMatrixD
ctypedef Matrix[adouble]* pMatrixAd
ctypedef map[block_key, Vector[double]]* pBlockMap

cdef extern from "inference_manager.h":
    cdef cppclass block_key:
        int& operator[](int)
    ctypedef vector[vector[double]] ParameterVector
    cdef cppclass InferenceManager nogil:
        InferenceManager(const int, const vector[int],
                const vector[int*], const vector[double],
                const double, const double) except +
        void setParams_d(const ParameterVector)
        void setParams_ad(const ParameterVector, vector[pair[int, int]] derivatives) 
        void Estep(bool)
        vector[double] loglik()
        vector[adouble] Q()
        vector[double] randomCoalTimes(const ParameterVector, double, int)
        double R(const ParameterVector, double t)
        bool debug
        bool saveGamma
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
    Matrix[T] sfs_cython[T](int, const ParameterVector&, double, double, double)
    Matrix[T] sfs_cython[T](int, const ParameterVector&, double, double, double, vector[pair[int, int]])


cdef extern from "piecewise_exponential_rate_function.h":
    cdef cppclass PiecewiseExponentialRateFunction[T]:
        PiecewiseExponentialRateFunction(const ParameterVector, const vector[double])
        T R(T x)
