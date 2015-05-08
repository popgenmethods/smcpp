from libcpp.vector cimport vector

cdef extern from "piecewise_exponential.h":
    cdef cppclass PiecewiseExponential:
        PiecewiseExponential(const vector[double]&,  const vector[double]&, const vector[double]&)
        double inverse_rate(double, double, double)

cdef extern from "conditioned_sfs.h":
    cdef cppclass ConditionedSFS:
        ConditionedSFS(PiecewiseExponential*, int)
        void compute(int, int, double*, vector[double*], int*, double*)
        void store_results(double*, double*)

cdef extern from "transition.h":
    cdef cppclass Transition:
        Transition(PiecewiseExponential*, const vector[double]&, double)
        void compute()
        void store_results(double*, double*)

# cdef extern from "hmm.h":
#     cdef cppclass HMM:
#         HMM(int _K, double* sqrt_a, double* b, double* sqrt_s, int _L, PyObject* _obs,
#                 int _M, int _n, PyObject* _pi, PyObject* _emission, PyObject* _transition,
#                 PyObject* _pi_jac, PyObject* _emission_jac, PyObject* _transition_jac)
#         double logp(double* jac)
#         vector[int]& viterbi()
