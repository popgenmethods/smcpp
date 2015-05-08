cdef extern from "matrix_exponential.h":
    void matrix_exponential(const int, const double*, double*)

