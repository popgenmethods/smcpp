from libcpp.vector cimport vector

cdef extern from "common.h":
    ctypedef struct AdMatrix:
        pass
    struct AdMatrixWrapper:
        pass

cdef extern from "piecewise_exponential.h":
    cdef cppclass PiecewiseExponential:
        PiecewiseExponential(const vector[double]&, const vector[double]&, const vector[double]&)
        double double_inverse_rate(double, double, double)
        double double_R(double)
        void print_debug()

cdef extern from "conditioned_sfs.h":
    AdMatrix calculate_sfs(PiecewiseExponential eta, int n, int S, int M, const vector[double] &ts, 
            const vector[double*] &expM, double t1, double t2, int numthreads, double theta)
    void store_sfs_results(const AdMatrix&, double*, double*)

cdef extern from "transition.h":
    cdef cppclass Transition:
        Transition(const PiecewiseExponential&, const vector[double]&, double)
        void compute()
        void store_results(double*, double*)

cdef extern from "hmm.h":
    double compute_hmm_likelihood(double*, const PiecewiseExponential &eta,
            const vector[AdMatrix]& emission, int L, const vector[int*] obs, 
            const vector[double] &hidden_states, const double rho, int numthreads,
            vector[vector[int]] &viterbi_paths, bint viterbi,
            double reg_a, double reg_b, double reg_s)
    # vector[int]& viterbi()
