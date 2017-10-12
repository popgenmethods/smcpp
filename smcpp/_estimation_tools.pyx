from libc.math cimport exp, log
import numpy as np


def thin_data(data, int thinning, int offset=0):
    '''
    Implement the thinning procedure needed to break up correlation
    among the full SFS emissions.
    '''
    # Thinning
    cdef int i = offset
    out = []
    cdef int[:, :] vdata = data
    bases = data[:, 0].sum()
    R = int((2 * np.ceil(data[offset:, 0] / thinning)).sum())
    ret = np.zeros([R, data.shape[1]], dtype=np.int32)
    cdef int j, n, v, span, sa
    cdef int K = data.shape[0]
    cdef int npop = (data.shape[1] - 1) / 3
    a = np.zeros(npop, dtype=np.int32)
    b = np.zeros_like(a)
    nb = np.zeros_like(a)
    thin = np.zeros(npop * 3, dtype=np.int32)
    nonseg = np.zeros(npop * 3, dtype=np.int32)
    cdef int[:] a_view = a, b_view = b, nb_view = nb, thin_view = thin, nonseg_view = nonseg
    cdef int[:, :] vret = ret
    cdef int r = 0
    with nogil:
        for j in range(K):
            span = vdata[j, 0]
            # a_view[:] = vdata[j, 1::3]
            # b_view[:] = vdata[j, 2::3]
            # nb_view[:] = vdata[j, 3::3]

            # sa = sum_memview(vdata[j, 1::3])
            # thin_view[::3] = a_view
            sa = 0
            for n in range(npop):
                v = vdata[j, 1 + 3 * n]
                sa += v
                thin_view[3 * n] = v
            if sa == 2:
                for n in range(npop):
                    thin_view[3 * n] = 0
            while span > 0:
                if i < thinning and i + span >= thinning:
                    if thinning - i > 1:
                        # out.append([thinning - i - 1] + list(thin_view))
                        vret[r, 0] = thinning - i - 1
                        for n in range(3 * npop):
                            vret[r, 1 + n] = thin_view[n]
                        r += 1
                    if sa == 2 and all_eq_memview(b_view, nb_view):
                        # nonseg_view[2::3] = nb_view
                        for n in range(npop):
                            nonseg_view[2 + 3 * n] = nb_view[n]
                        # out.append([1] + list(nonseg_view))
                        vret[r, 0] = 1
                        for n in range(3 * npop):
                            vret[r, 1 + n] = nonseg_view[n]
                        r += 1
                    else:
                        # out.append([1] + list(vdata[j, 1:]))
                        vret[r, 0] = 1
                        for n in range(3 * npop):
                            vret[r, 1 + n] = vdata[j, 1 + n]
                        r += 1
                    span -= thinning - i
                    i = 0
                else:
                    # out.append([span] + list(thin_view))
                    vret[r, 0] = span
                    for n in range(3 * npop):
                        vret[r, 1 + n] = thin_view[n]
                    r += 1
                    i += span
                    break
    ret = ret[:r]
    # ret = np.array(out, dtype=np.int32)
    assert ret[:, 0].sum() == data[:, 0].sum()
    return ret


cdef bint all_eq_memview(int[:] a, int[:] b) nogil:
    for i in range(a.shape[0]):
        if a[i] != b[i]:
            return False
    return True


cdef int sum_memview(int[:] a) nogil:
    cdef int ret = 0
    for i in range(a.shape[0]):
        ret += a[i]
    return ret


# @cython.boundscheck(False)
def realign(contig, int w):
    'Realign contig data to have a split every w bps'
    assert w > 0
    cdef int[:, :] cd = contig.data
    cdef int n = cd.shape[0], m = cd.shape[1]
    L = np.ceil(1 + contig.data[:, 0] / w).sum().astype("int")
    ret = np.zeros([L + 1, m], dtype=np.int32)
    cdef int[:, :] vret = ret
    cdef int[:] last = np.zeros(m, dtype=np.int32)
    cdef int i = 0, j = 0, k, seen = 0, r
    last[:] = cd[0]
    with nogil:
        while True:
            j += 1
            for k in range(m):
                vret[j - 1, k] = last[k]
            if seen + last[0] > w:
                r = w - seen
                vret[j - 1, 0] = r
                last[0] -= r
                seen = 0
            else:
                seen += last[0]
                i += 1
                if i == n:
                    break
                for k in range(m):
                    last[k] = cd[i, k]
    ret[j] = last
    ret = ret[:j]
    ret = ret[ret[:, 0] > 0]
    assert np.all(ret[:, 0].sum(axis=0) == contig.data[:, 0].sum(axis=0))
    assert np.all(ret[:, 0] > 0)
    contig.data = ret
