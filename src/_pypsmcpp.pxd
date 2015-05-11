# from libcpp.vector cimport vector
from libcpp.map cimport map

cdef extern from "<vector>" namespace "std":
    cdef cppclass vector[T]:
        cppclass iterator:
            T operator*()
            iterator operator++()
            bint operator==(iterator)
            bint operator!=(iterator)
        vector()
        void push_back(T&)
        void emplace_back()
        T& operator[](int)
        T& at(int)
        iterator begin()
        iterator end()
        

# Necessary to work around a bug
ctypedef ConditionedSFS* ConditionedSFSptr

cdef extern from "piecewise_exponential.h":
    cdef cppclass PiecewiseExponential:
        PiecewiseExponential(const vector[double]&,  const vector[double]&, const vector[double]&)
        double double_inverse_rate(double, double, double)
        void print_debug()

cdef extern from "conditioned_sfs.h":
    cdef cppclass ConditionedSFS:
        ConditionedSFS(PiecewiseExponential*, int)
        void compute(int, int, double*, vector[double*], int*, double*)
        void store_results(double*, double*)
        void set_seed(int)

cdef extern from "transition.h":
    cdef cppclass Transition:
        Transition(PiecewiseExponential*, const vector[double]&, double)
        void compute()
        void store_results(double*, double*)

cdef extern from "hmm.h":
    cdef cppclass HMM:
        HMM(PiecewiseExponential *eta, 
            vector[vector[ConditionedSFSptr]] &csfs,
            const vector[double] hidden_states,
            int L, int* obs, double rho, double theta)
        double logp(double* jac)
        # vector[int]& viterbi()
