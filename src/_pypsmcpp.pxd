from libcpp.vector cimport vector

cdef extern from "common.h":
    ctypedef struct adouble:
        pass
    ctypedef struct DoubleVector:
        pass
    ctypedef struct DoubleMatrix:
        pass
    cdef double toDouble(const adouble &)

cdef extern from "conditioned_sfs.h":
    cdef cppclass MatrixInterpolator:
        MatrixInterpolator(int, vector[double], vector[double*])

    void init_eigen()
    void set_seed(long long)
    void cython_calculate_sfs(const vector[vector[double]] &params,
            int n, int num_samples, const MatrixInterpolator&,
            double tau1, double tau2, int numthreads, double theta, 
            double* outsfs)
    void cython_calculate_sfs_jac(const vector[vector[double]] &params,
            int n, int num_samples, const MatrixInterpolator&,
            double tau1, double tau2, int numthreads, double theta, 
            double* outsfs, double* outjac)

cdef extern from "transition.h":
    void cython_calculate_transition(const vector[vector[double]] &params,
            const vector[double] hidden_states, double rho, double* outtrans)
    void cython_calculate_transition_jac(const vector[vector[double]] &params,
            const vector[double] hidden_states, double rho, double* outtrans, double* outjac)

cdef extern from "loglik.h":
    T sfs_loglik[T](
            const vector[vector[double]]&,
            const int,
            const int, 
            const MatrixInterpolator &,
            double*,
            int,
            double, double)

#    T loglik[T](
#            const vector[vector[double]]&,
#            const int, 
#            const int, const int,
#            const vector[double]&, const vector[double*]&,
#            const int, const vector[int*], 
#            const vector[double]&,
#            const double, const double,
#            const int,
#            int,
#            bool, vector[vector[int]]&,
#            double)

    T compute_Q[T](
            const vector[vector[double]]&,
            const int, const int,
            const MatrixInterpolator &,
            const int, const vector[int*], 
            const vector[double]&,
            const double, const double,
            const int,
            int,
            double,
            vector[DoubleMatrix] &, 
            vector[DoubleMatrix] &, 
            bint)

    void fill_jacobian(const adouble &, double*)
