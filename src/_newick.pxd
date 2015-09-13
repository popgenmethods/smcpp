from libcpp.string cimport string

cdef extern from "newick.h":
    double cython_tmrca(const string, const string, const string)
