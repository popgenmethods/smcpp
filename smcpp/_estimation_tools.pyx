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
    cdef int i
    for i in range(a.shape[0]):
        if a[i] != b[i]:
            return False
    return True

cdef bint any_pos_memview(int[:] a) nogil:
    cdef int i
    for i in range(a.shape[0]):
        if a[i] > 0:
            return True
    return False

cdef void copy_memview(int[:] a, int[:] b) nogil:
    cdef int i
    for i in range(a.shape[0]):
        a[i] = b[i]

cdef int sum_memview(int[:] a) nogil:
    cdef int ret = 0
    for i in range(a.shape[0]):
        ret += a[i]
    return ret


cdef void process_bin(int[:, :] data, int[:, :] new_data, long[:] na,
                      int i, int j, int k, bint thin) nogil:
    cdef int q, aa, bb, sample_size, max_sample_size, mq
    max_sample_size = -2
    cdef int K, seg
    K = (data.shape[1] - 1) // 3
    mq = 0
    for q in range(i, j + 1):
        seg = 0
        if data[q, 0] == 0:
            continue
        sample_size = 0
        for aa in range(K):
            bb = 3 * aa
            if thin:
                sample_size += data[q, bb + 1]
            else:
                sample_size += data[q, bb + 3]
                sample_size += na[aa] * (data[q, bb + 1] >= 0)
            seg += max(0, data[q, bb + 1])
        if sample_size > max_sample_size:
            mq = q
            max_sample_size = sample_size
        if max_sample_size == 2 and seg == 1:
            mq = q
    for aa in range(K):
        bb = 3 * aa
        new_data[k, bb + 1] = data[mq, bb + 1] 
        if not thin:
            new_data[k, bb + 2] = data[mq, bb + 2]
            new_data[k, bb + 3] = data[mq, bb + 3]


def bin_observations(contig, long w):
    cdef int[:, :] data = contig.data
    cdef long[:] na = contig.a
    cdef int i, j, k, span, seen, bb
    cdef int[:] row, best_row, next_row
    ret = np.zeros([len(contig) // w + 1, data.shape[1]], dtype=np.int32)
    cdef int[:, :] new_data = ret
    with nogil:
        i = 0
        j = 0
        k = 0
        seen = 0
        while j < data.shape[0]:
            span = data[j, 0]
            if seen + span > w:
                data[j, 0] = w - seen
                process_bin(data, new_data, na, i, j, k, 0)
                data[j, 0] = span - (w - seen)
                seen = 0
                k += 1
                i = j
            else:
                j += 1
                seen += span
    process_bin(data, new_data, na, i, j - 1, k, 0)
    ret[:, 0] = 1
    return ret[:k + 1]


# @cython.boundscheck(False)
def realign(data, int w):
    'Realign contig data to have a split every w bps'
    assert w > 0
    cdef int[:, :] cd = data
    cdef int n = cd.shape[0], m = cd.shape[1]
    L = np.ceil(1 + data[:, 0] / w).sum().astype("int")
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
    assert np.all(ret[:, 0].sum(axis=0) == data[:, 0].sum(axis=0))
    assert np.all(ret[:, 0] > 0)
    return ret


def windowed_mutations_helper(contig, int w):
    assert w > 0
    cdef int[:, :] cd = contig.data[::-1]
    cdef int seen, nmiss, mut, sp, span, extra, k
    L = contig.data[:, 0].sum()
    ret = np.zeros([L // w + 1, 2], dtype=np.int32)
    cdef int n = (contig.data.shape[1] - 1) // 3
    cdef int[:, :] vret = ret
    cdef int i = cd.shape[0] - 1
    cdef int j = 0
    cdef int a = 0
    cdef int[:] last = np.zeros(1 + 3 * n, dtype=np.int32)
    last[:] = cd[i]
    seen = nmiss = mut = 0
    with nogil:
        while i >= 0:
            span = last[0]
            sp = span
            if w - seen < span:
                sp = w - seen
            extra = seen + span - w
            seen += sp
            a = 0
            for k in range(n):
                if last[1 + 3 * k] != -1:
                    a += last[1 + 3 * k]
                else:
                    a = -1
                    break
            if a >= 0:
                mut += sp * (a % 2)
                nmiss += sp
            if extra > 0:
                last[0] = extra
                vret[j, 0] = nmiss
                vret[j, 1] = mut
                j += 1
                nmiss = mut = seen = 0
            else:
                i -= 1
                for k in range(last.shape[0]):
                    last[k] = cd[i, k]
    ret[j] = [nmiss, mut]
    return ret
