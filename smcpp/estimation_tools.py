'Miscellaneous estimation and data-massaging functions.'
from __future__ import absolute_import, division, print_function
import numpy as np
from logging import getLogger
import scipy.optimize
import scipy.interpolate
from concurrent.futures import ProcessPoolExecutor
from collections import namedtuple
import json
import pandas as pd
import itertools

from . import _smcpp, util
from .contig import Contig


logger = getLogger(__name__)


# Construct time intervals stuff
def extract_pieces(piece_str):
    '''Convert PSMC-style piece string to model representation.'''
    pieces = []
    for piece in piece_str.split("+"):
        try:
            num, span = list(map(int, piece.split("*")))
        except ValueError:
            span = int(piece)
            num = 1
        pieces += [span] * num
    return pieces


def construct_time_points(t1, tK, pieces, offset):
    s = np.diff(np.logspace(np.log10(offset + t1),
                            np.log10(tK),
                            sum(pieces) + 1))
    time_points = np.zeros(len(pieces))
    count = 0
    for i, p in enumerate(pieces):
        time_points[i] = s[count:(count + p)].sum()
        count += p
    return np.concatenate([[t1], time_points])


def compress_repeated_obs(dataset):
    # pad with illegal value at starting position
    nonce = np.zeros_like(dataset[0])
    nonce[:2] = [1, -999]
    dataset = np.r_[[nonce], dataset, [nonce]]
    nonreps = np.any(dataset[1:, 1:] != dataset[:-1, 1:], axis=1)
    newob = dataset[1:][nonreps]
    csw = np.cumsum(dataset[:, 0])[np.where(nonreps)]
    newob[:-1, 0] = csw[1:] - csw[:-1]
    return newob[:-1]


def _thin_helper(args):
    thinned = np.array(_smcpp.thin_data(*args), dtype=np.int32)
    return compress_repeated_obs(thinned)


def thin_dataset(dataset, thinning):
    '''Only emit full SFS every <thinning> sites'''
    with ProcessPoolExecutor() as p:
        return list(p.map(_thin_helper, 
            [(chrom, th, i) for i, (chrom, th) in enumerate(zip(dataset, thinning))]))


def recode_nonseg(contigs, cutoff):
    warn_only = False
    if cutoff is None:
        cutoff = 100000
        warn_only = True
    for c in contigs:
        d = c.data
        runs = (
            (d[:, 0] > cutoff) &
            np.all(d[:, 1::3] == 0, axis=1) &
            np.all(d[:, 2::3] == 0, axis=1)
        )
        if np.any(runs):
            if warn_only:
                f = logger.warning
                txt = ""
            else:
                f = logger.debug
                txt = " (converted to missing)"
                d[runs, 1::3] = -1
                d[runs, 3::3] = 0
            f("Long runs of homozygosity%s in contig %s: \n%s", txt, c.fn, d[runs])
    return contigs


def break_long_spans(contigs, length_cutoff):
    # Spans longer than this are broken up
    # FIXME: should depend on rho
    span_cutoff = 1000000
    contig_list = []
    obs_attributes = {}
    for i, contig in enumerate(contigs):
        obs = contig.data
        miss = np.zeros_like(obs[0])
        miss[0] = 1
        miss[1::3] = -1
        long_spans = np.where(
            (obs[:, 0] >= span_cutoff) &
            np.all(obs[:, 1::3] == -1, axis=1) &
            np.all(obs[:, 3::3] == 0, axis=1))[0]
        cob = 0
        if obs[long_spans].size:
            logger.debug("Long missing spans: \n%s", (obs[long_spans]))
        positions = np.insert(np.cumsum(obs[:, 0]), 0, 0)
        for x in long_spans.tolist() + [None]:
            s = obs[cob:x, 0].sum()
            if s > length_cutoff:
                contig_list.append(Contig(data=np.insert(obs[cob:x], 0, miss, 0),
                                          pid=contig.pid, fn=contig.fn, n=contig.n, a=contig.a))
                if contig.a[0] == 1:
                    a_cols = [1, 4]
                else:
                    assert contig.a[0] == 2
                    a_cols = [1]
                last_data = contig_list[-1].data
                l = last_data[:, 0].sum()
                lda = last_data[:, a_cols]
                s2 = lda[lda.min(axis=1) >= 0].sum()
                assert s2 >= 0
                obs_attributes.setdefault(i, []).append(
                    (positions[cob],
                     positions[x] if x is not None else positions[-1],
                     l, 1. * s2 / l))
            else:
                if s > 0:
                    logger.debug("omitting sequence of length %d "
                                "as < length cutoff %d" %
                                (s, length_cutoff))
            try:
                cob = x + 1
            except TypeError:  # fails for final x=None
                pass
    return contig_list, obs_attributes


def balance_hidden_states(model, M):
    """
    Return break points [0, b_1, ..., b_M, oo) such that
    the probability of coalescing in each interval under the
    model is the same. (Breaks are returned in units of
    generations.)

    """
    M -= 1
    eta = _smcpp.PyRateFunction(model, [])
    ret = [0.0]
    # ms = np.arange(0.1, 2.1, .1).tolist() + list(range(3, M))
    ms = range(1, M)
    for m in ms:
        def f(t):
            Rt = float(eta.R(t))
            return np.exp(-Rt) - 1.0 * (M - m) / M
        res = scipy.optimize.brentq(f, ret[-1], 1000.)
        ret.append(res)
    ret.append(np.inf)
    return np.array(ret) * 2 * model.N0  # return in generations


def watterson_estimator(contigs):
    with ProcessPoolExecutor() as p:
        num = denom = 0
        for S, sample_sizes, spans in map(_watterson_helper, contigs):
            num += S
            non_missing = sample_sizes > 0
            ss = sample_sizes[non_missing]
            sp = spans[non_missing]
            denom += (sp * (np.log(ss) + 0.5 / ss + 0.57721)).sum()
    return num / denom


def _watterson_helper(contig):
    c = contig
    shp = [x + 1 for na in zip(c.a, c.n) for x in na]
    ret = np.zeros(shp, dtype=int)
    spans = c.data[:, 0]
    seg = (np.any(c.data[:, 1::3] >= 1, axis=1) |
           np.any(c.data[:, 2::3] >  0, axis=1))
    S = spans[seg].sum()
    sample_sizes = (c.data[:, 3::3].sum(axis=1) +
                    np.maximum(0, c.data[:, 1::3]).sum(axis=1))
    return (S, sample_sizes, spans)

def _load_data_helper(fn):
    try:
        # This parser is way faster than np.loadtxt
        A = pd.read_csv(fn, sep=' ', comment="#", header=None).values
    except ImportError as e:
        logger.debug(e)
        A = np.loadtxt(fn, dtype=np.int32)
    except:
        logger.error("In file %s", fn)
        raise
    if len(A) == 0:
        raise RuntimeError("empty dataset: %s" % fn)
    with util.optional_gzip(fn, "rt") as f:
        first_line = next(f).strip()
        if first_line.startswith("# SMC++"):
            attrs = json.loads(first_line[7:])
            a = [len(a) for a in attrs['dist']]
            n = [len(u) for u in attrs['undist']]
            if "pids" not in attrs:
                logger.warn("%s lacks population ids. Timidly proceeding with defaults...", fn)
                attrs["pids"] = ["pop%d" % i for i, _ in enumerate(a, 1)]
        else:
            logger.warn("File %s doesn't appear to be in SMC++ format", fn)
            npop = (A.shape[1] - 1) // 3
            attrs = {'pids': ['pop%d' % i for i in range(1, npop + 1)]}
            a = A[:, 1::3].max(axis=0)
            n = A[:, 3::3].max(axis=0)
    pid = tuple(attrs['pids'])
    # Internally we always put the population with the distinguished lineage first.
    if len(a) == 2 and a[0] == 0 and a[1] == 2:
        n = n[::-1]
        a = a[::-1]
        pid = pid[::-1]
        A = A[:, [0, 4, 5, 6, 1, 2, 3]]
    data = np.ascontiguousarray(A, dtype='int32')
    return Contig(pid=pid, data=data, n=n, a=a, fn=fn)


def files_from_command_line_args(args):
    ret = []
    for f in args:
        if f[0] == "@":
            ret += [line.strip() for line in open(f[1:], "rt") if line.strip()]
        else:
            ret.append(f)
    return set(ret)


def load_data(files):
    with ProcessPoolExecutor() as p:
        obs = list(p.map(_load_data_helper, files))
    return obs


def windowed_mutations(contigs, w):
    '''Return array [[window_length, num_mutations], ...] for each contig'''
    with ProcessPoolExecutor() as p:
        return list(p.map(_windowed_mutations_helper, contigs, itertools.repeat(w)))


def _windowed_mutations_helper(*args):
    contig, w = args
    assert w > 0
    cd = contig.data[::-1]
    seen = nmiss = mut = 0
    ret = []
    i = cd.shape[0] - 1
    last = cd[i].tolist()
    while i >= 0:
        span, *abnb = last
        a = abnb[::3]
        sp = min(w - seen, span)
        extra = seen + span - w
        seen += sp
        if -1 not in a:
            mut += sp * (sum(abnb[::3]) % 2)
            nmiss += sp
        if extra > 0:
            last = [extra] + abnb
            ret.append([nmiss, mut])
            nmiss = mut = seen = 0
        else:
            i -= 1
            last = cd[i].tolist()
    ret.append([nmiss, mut])
    return ret
