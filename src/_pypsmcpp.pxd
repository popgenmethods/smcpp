from libcpp.vector cimport vector

cdef extern from "common.h":
    ctypedef struct adouble:
        pass
    cdef double toDouble(const adouble &)

cdef extern from "conditioned_sfs.h":
    void set_seed(unsigned int)

cdef extern from "loglik.h":
    T loglik[T](
            const vector[double]&, const vector[double]&, const vector[double]&,
            const int, 
            const int, const int,
            const vector[double]&, const vector[double*]&,
            const int, const vector[int*], 
            const vector[double]&,
            const double, const double,
            int,
            bool, vector[vector[int]]&,
            double, double, double)
    void fill_jacobian(const adouble &, double*)

