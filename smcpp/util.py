from __future__ import absolute_import, division, print_function
from contextlib import contextmanager
import gzip
import numpy as np
import functools
import multiprocessing
import os
from itertools import zip_longest

import logging
logger = logging.getLogger(__name__)


def cumsum0(ary):
    return np.concatenate([[0], np.cumsum(ary)])


# Section 7. of MSMC supplemental
def build_sawtooth():
    sawtooth = {"a": [5.], "b": [], "s": []}
    g_last = t_last = 0.
    sawtooth_events = [
        (.000582262, 1318.18),
        (.00232905, -329.546),
        (.00931919, 82.3865),
        (.0372648, -20.5966),
        (.149059, 5.14916),
        (0.596236, 0.),
    ]
    for t, g in sawtooth_events:
        sawtooth["b"].append(sawtooth["a"][-1] * np.exp(g_last * (t_last - t)))
        sawtooth["a"].append(sawtooth["b"][-1])
        sawtooth["s"].append(t - t_last)
        g_last = g
        t_last = t
    sawtooth["b"].append(sawtooth_events[-1][0])
    sawtooth["s"].append(.1)
    sawtooth = {k: np.array(sawtooth[k]) for k in sawtooth}
    sawtooth["s"] *= 2.
    sawtooth["N0"] = 14312
    return sawtooth


sawtooth = build_sawtooth()

human = {
    "a": np.array([10.0, 0.5, 1.0, 4.0]),
    "b": np.array([1.0, 0.5, 1.0, 4.0]),
    "s": np.array([10000., 70000. - 10000., 200000. - 70000., 1.0]) / 20000. / 29.0,
    "N0": 10000.,
}


def undistinguished_sfs(sfs, folded=False):
    a = sfs.shape[0] - 1
    n = sfs.shape[1] - 1
    new_shape = [n + a] + list(sfs.shape[2:])
    usfs = np.zeros(new_shape, dtype=sfs.dtype)
    for i in range(a + 1):
        for j in range(n + 1):
            if 0 <= i + j < n + a:
                usfs[i + j] = usfs[i + j] + sfs[i, j]
    if folded:
        ret = [usfs[0]]
        for i in range(1, -(-(n + a) // 2)):
            inds = list({i, n + a - i})
            ret.append(usfs[inds].sum(axis=0))
        usfs = np.array(ret, dtype=sfs.dtype)
    return usfs


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)


def unpack(iterable):
    for span, x in iterable:
        for i in range(span):
            yield x


def pack(seq):
    iterable = iter(seq)
    x = next(iterable)
    i = 1
    for xp in iterable:
        if xp == x:
            i += 1
        else:
            yield (i, x)
            x = xp
            i = 1
    yield (i, x)


def memoize(obj):
    cache = obj.cache = {}

    @functools.wraps(obj)
    def memoizer(*args, **kwargs):
        key = tuple(args) + tuple(kwargs.items())
        if key not in cache:
            cache[key] = obj(*args, **kwargs)
        return cache[key]

    return memoizer


def kl(sfs1, sfs2):
    s1 = sfs1.flatten()
    s2 = sfs2.flatten()
    nz = s1 != 0.0
    return (s1[nz] * (np.log(s1[nz]) - np.log(s2[nz]))).sum()


def dataset_from_panel(dataset, n, distinguished_rows, random=True):
    L, positions, haps = dataset[:3]
    dr = list(distinguished_rows)
    K = haps.shape[1]
    if n < haps.shape[0]:
        panel = haps[[i for i in range(haps.shape[0]) if i not in dr]]
        N, K = panel.shape
        h2 = np.zeros([n, K], dtype=np.int8)
        h2[:2] = haps[dr]
        if n > 2:
            for i in range(K):
                inds = [j for j in range(N) if j not in dr and panel[j, i] != -1]
                if random:
                    inds = np.random.choice(inds, n - 2, replace=False)
                assert len(inds) >= n - 2
                h2[2:, i] = panel[inds, i]
        haps = h2
    else:
        perm = np.arange(n)
        perm[[0, 1] + dr] = perm[dr + [0, 1]]
        haps = haps[np.ix_(perm, np.arange(K))]
    seg = np.logical_and(*[(haps != a).sum(axis=0) > 0 for a in [0, 1]])
    return (L, positions[seg], haps[:, seg]) + dataset[3:]


def hmm_data_format(dataset, n, distinguished_rows):
    # Convert a dataset generated by simulate() to the format accepted
    # by the inference code
    ret = []
    p = 0
    L, positions, haps = dataset_from_panel(dataset, n, distinguished_rows)[:3]
    d = haps[:2].sum(axis=0)
    d[haps[:2].min(axis=0) == -1] = -1
    t = np.maximum(0, haps[2:]).sum(axis=0)
    en = (haps[2:] != -1).sum(axis=0)
    nd = d.shape[0]
    nrow = 2 * nd - 1
    ret = np.zeros([nrow, 4], dtype=np.int32)
    ret[::2, 0] = 1
    ret[::2, 1] = d
    ret[::2, 2] = t
    ret[::2, 3] = en
    gaps = positions[1:] - positions[:-1] - 1
    ret[1::2, 0] = gaps
    ret[1::2, 1:3] = 0
    ret[1::2, 3] = n - 2
    if positions[0] > 0:
        ret = np.vstack(([positions[0], 0, 0, n - 2], ret))
    if positions[-1] < L - 1:
        ret = np.vstack((ret, [L - 1 - positions[-1], 0, 0, n - 2]))
    # eliminate "no gaps"
    ret = ret[ret[:, 0] > 0]
    # assert np.all(ret >= 0)
    assert ret.sum(axis=0)[0] == L, (L, ret.sum(axis=0)[0], ret)
    assert np.all(ret[:, 0] >= 1)
    return ret


def break_long_missing_spans(data, span_cutoff=50000):
    ret = [[]]
    inds = np.where(data[:, 0] >= span_cutoff)
    lastobs = data[0]
    for obs in data[1:]:
        if obs[0] == 0:
            continue
        if np.all(obs[1:] == lastobs[1:]):
            lastobs[0] += obs[0]
        else:
            if lastobs[0] > span_cutoff:
                logger.debug("Skipping long span: %s" % str(lastobs))
                ret.append([])
            else:
                ret[-1].append(lastobs)
            lastobs = obs
    ret[-1].append(lastobs)
    r2 = []
    for rr in ret:
        if rr == []:
            continue
        if rr[0][0] > 1:
            rr.insert(0, np.concatenate([[1], rr[0][1:]]))
            rr[1][0] -= 1
        r2.append(np.array(rr, dtype=np.int32))
    return r2


@contextmanager
def optional_gzip(f, mode):
    with gzip.open(f, mode) if f.endswith(".gz") else open(f, mode) as o:
        yield o


class RepeatingWriter:

    def __init__(self, f):
        self.f = f
        self.last_ob = None
        self.i = 0

    def write(self, ob):
        if self.last_ob is None:
            self.last_ob = ob
            return
        if ob[1:] == self.last_ob[1:]:
            self.last_ob[0] += ob[0]
        else:
            self._write_last_ob()
            self.last_ob = ob

    def _write_last_ob(self):
        if self.last_ob is not None and self.last_ob[0] > 0:
            fmtstr = " ".join(["%d"] * len(self.last_ob)) + "\n"
            self.f.write(fmtstr % tuple(self.last_ob))
            self.i += 1

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self._write_last_ob()
        self.f = None
        logger.info("Wrote %d observations" % self.i)


def format_ad(ad_x):
    "String representation of an autodiff number"
    FMT = "%.1f"
    if isinstance(ad_x, float):
        return FMT % ad_x
    dx = sorted([key for key in ad_x.d() if key.tag is not None], key=lambda x: x.tag)
    deriv = [FMT % ad_x.d(y) for y in dx]
    return (FMT + " [%s]") % (ad_x.x, ", ".join(deriv))
