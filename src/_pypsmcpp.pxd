from libcpp.vector cimport vector

cdef extern from "common.h":
    ctypedef struct adouble:
        pass
    cdef double toDouble(const adouble &)

cdef extern from "conditioned_sfs.h":
    void set_seed(long long)
    void cython_calculate_sfs(const vector[double] diff_x, const vector[double] sqrt_y,
            int n, int S, int M, const vector[double] &ts, 
            const vector[double*] &expM, double tau1, double tau2, int numthreads, double theta, 
            double* outsfs)
    void cython_calculate_sfs_jac(const vector[double] diff_x, const vector[double] sqrt_y,
            int n, int S, int M, const vector[double] &ts, 
            const vector[double*] &expM, double tau1, double tau2, int numthreads, double theta, 
            double* outsfs, double* outjac)

cdef extern from "transition.h":
    void cython_calculate_transition(const vector[double] &diff_x, const vector[double] &sqrt_y,
            const vector[double] hidden_states, double rho, double* outtrans)
    void cython_calculate_transition_jac(const vector[double] &diff_x, const vector[double] &sqrt_y,
            const vector[double] hidden_states, double rho, double* outtrans, double* outjac)

cdef extern from "loglik.h":
    T loglik[T](
            const vector[double]&, const vector[double]&, 
            const int, 
            const int, const int,
            const vector[double]&, const vector[double*]&,
            const int, const vector[int*], 
            const vector[double]&,
            const double, const double,
            int,
            bool, vector[vector[int]]&,
            double) except +
    void fill_jacobian(const adouble &, double*)

