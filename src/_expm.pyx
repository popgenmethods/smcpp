cimport numpy as np
import numpy as np

def expm(np.ndarray[double, ndim=2, mode="c"] M):
    n = M.shape[0]
    assert M.shape[1] == n 
    ret = np.ascontiguousarray(np.zeros([n, n], dtype=np.double, order="C"))
    cdef double[:, ::1] Mv = M
    cdef double[:, ::1] retv = ret
    matrix_exponential(n, &Mv[0, 0], &retv[0, 0])
    return ret
