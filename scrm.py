from __future__ import division
import bitarray
import numpy as np
import math
import os
import sh
from subprocess import check_output
import subprocess
import itertools
import logging
import sys

from Bio import Phylo
from cStringIO import StringIO
import networkx as nx

import util

logger = logging.getLogger(__name__)
scrm = sh.Command(os.environ['SCRM_PATH'])

# Stuff related to posterior decoding
# Format trees
def tree_obs_iter(l1, l2, trees):
    fs = frozenset([l1, l2])
    for sec, d in trees:
        for i in range(sec):
            yield d[fs]

def true_hidden_states(trees, distinguished_lineages):
    tb = []
    M = len(hs) - 1
    for block in util.grouper(tree_obs_iter(trees), block_size):
        a = np.zeros([M, 1])
        c = Counter(block)
        s = sum(c.values())
        for k in c:
            ip = np.searchsorted(hs, k) - 1
            a[ip] = 1. * c[k] / s
        tb.append(a)
    return np.array(tb).T

def splitter(_iter, key="//"):
    def f(line):
        if line.strip() == key:
            f.i += 1
        return f.i
    f.i = 0
    for (k, subiter) in itertools.groupby(_iter, f):
        yield (k, (line for line in subiter if line.strip() != key))

def demography_from_params(params):
    demography = []
    ct = 0.0
    z = list(zip(*params))
    for ai, bi, si in z[:-1]:
        beta = (np.log(ai * 2.0) - np.log(bi * 2.0)) / (si * 2.0)
        demography += ['-eN', ct, ai * 2.0]
        if beta != 0.0:
            demography += ['-eG', ct, beta]
        ct += si * 2.0
    demography += ['-eN', ct, z[-1][0]]
    return demography

def print_data_stats(positions, haps):
    gaps = positions[1:] - positions[:-1]
    print("Minimum gap: %d" % np.min(gaps))
    print("Average gap: %d" % np.mean(gaps))
    print("# seg. sites: %d" % gaps.shape[0])
    i = np.argmin(gaps)
    print(positions[(i-3):(i+3)])

def newick_to_dists(newick, leaves=None):
    tree = Phylo.to_networkx(Phylo.read(StringIO(newick), "newick"))
    ldict = {node.name: node for node in tree.nodes() if node.name}
    if leaves:
        ldict = {k: v for k, v in ldict.items() if k in leaves}
    dsts = {}
    all_dsts = nx.shortest_path_length(tree, weight='weight')
    for n1, n2 in itertools.combinations(ldict, 2):
        dsts[frozenset([int(n1) - 1, int(n2) - 1])] = all_dsts[ldict[n1]][ldict[n2]] / 2.0
    return dsts

def parse_scrm(n, L, output, include_trees):
    coal_times = []
    ts = 0
    for line in output:
        if line.startswith("segsites"):
            break
        if not include_trees:
            continue
        l = line.strip()
        k = l.index("(") - 1
        span = int(l[1:k])
        ts += span
        dsts = newick_to_dists(l[k:], include_trees)
        coal_times.append((span, dsts))
    positions = next(output).strip()
    if positions:
        positions = (L * np.array([float(x) for x in positions.split(" ")[1:]])).astype('int')
        # ignore trailing newline
        haps = [bitarray.bitarray(str(line).strip()) for line in output if line.strip()] 
        ret = (L, positions, haps)
        if include_trees:
            ret += (coal_times,)
        return ret
    return None

def simulate(n, N0, theta, rho, L, demography=[], include_coalescence_times=False):
    r = 4 * N0 * rho * (L - 1)
    t = 4 * N0 * theta * L
    args = [n, 1, '-p', int(math.log10(L)) + 1, '-t', t, '-r', r, L, '-l', 
            10000, '-seed', np.random.randint(0, sys.maxint)] + demography
    if include_coalescence_times:
        args.append("-T")
    output = scrm(*args, _iter=True)
    cmd_line, seed, _, _ = [line.strip() for line in itertools.islice(output, 4)]
    return parse_scrm(n, L, output, include_coalescence_times)

def distinguished_sfs(n, M, N0, theta, demography, t0=0.0, t1=np.inf):
    t = 4 * N0 * theta
    args = [n, M, '-t', t] + demography
    if t0 > 0.0 or t1 < np.inf:
        args.append("-T")
    cmd = os.environ['SCRM_PATH'] + " " + " ".join(map(str, args))
    output = scrm(*args, _iter=True)
    avgsfs = np.zeros([3, n - 1], dtype=float)
    fs = frozenset([0, 1])
    m = 0
    for k, lines in splitter(output):
        if k == 0:
            continue
        sfs = np.zeros([3, n - 1], dtype=float)
        if t0 > 0.0 or t1 < np.inf:
            newick = next(lines)
            d12 = newick_to_dists(newick, "12")[fs]
            if not t0 < d12 < t1:
                continue
        m += 1
        ss = next(lines)
        segsites = int(ss.strip().split(" ")[1])
        if segsites == 0:
            avgsfs[0, 0] += 1
            continue
        next(lines) # positions
        bits = np.array([[int(x) for x in line.strip()] for line in lines if line.strip()])
        assert (n, segsites) == bits.shape
        for col in range(segsites):
            sfs[bits[:2, col].sum(), bits[2:, col].sum()] += 1
        avgsfs += sfs
    print(m, M)
    avgsfs /= m
    return avgsfs

def sfs(n, M, N0, theta, demography):
    t = 4 * N0 * theta
    args = [n, M, '-t', t, '-oSFS'] + demography
    cmd = os.environ['SCRM_PATH'] + " " + " ".join(map(str, args))
    output = check_output(
            """%s | grep SFS | tail -n+2 | cut -f2- -d' ' | Rscript -e 'cat(colSums(read.table(file("stdin"))))'""" % cmd, shell=True)
    ret = np.array([float(x) for x in output.split()])
    assert ret.shape == (n - 1,)
    ret = np.append([M - ret.sum(),], ret)
    assert ret.sum() == M
    return ret

def hmm_data_format(dataset, distinguished_cols):
    # Convert a dataset generated by simulate() to 
    # the format accepted by the inference code
    ret = []
    p = 0
    L, positions, haps = dataset[:3]
    for i, pos in enumerate(positions):
        pp = pos - p
        if pp == 0:
            logger.warn("Tri-allelic site at position %d; ignoring" % pos)
            continue
        if pp > 1:
            ret.append([pp - 1, 0, 0])
        d = sum([haps[c][i] for c in distinguished_cols])
        t = sum([h[i] for h in haps])
        ret.append([1, d, t - d])
        p = pos
    if L > pos:
        ret.append([L - pos, 0, 0])
    return np.array(ret, dtype=np.int32)

if __name__ == "__main__":
    L = 1000000
    data = simulate(50, 10000.0, 1e-8, 1e-8, L, ['-n', 1, 1])
    hmmd = hmm_data_format(data, (0, 1))
