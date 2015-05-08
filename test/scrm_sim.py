#!/usr/bin/env python2.7
from __future__ import print_function, division
from subprocess import check_output
from math import log
import random
import numpy as np
import itertools
import os
import json
from collections import Counter, namedtuple, defaultdict

from phylogenies import leaves, newick2tree, parent_of
from sum_product import NodeState

SCRM_PATH = os.environ['SCRM_PATH']

def build_command_line(root, L, lineages_per_taxon):
    '''Given a tree, build a scrm command line which will simulate from it.'''
    ejopts = []
    Iopts = []
    enopts = []
    N0 = root.N
    # tfac = 1.0 / 4.0 / N0 / 25.0
    # Times are already scaled in Jack's implementation
    tfac = 1.0 
    # rho = 1e-9 * 4 * N0 * (L - 1)
    theta = 10.

    lineages = []
    lineage_map = {}
    for i, leaf_node in list(enumerate(leaves(root), 1)):
        nsamp = lineages_per_taxon
        Iopts.append(nsamp)
        lineage_map[leaf_node] = i
        lineages += [leaf_node.leaf_name] * nsamp
        age = leaf_node.edge_length * tfac
        enopts.append((age, i, leaf_node.N / N0))
        p = parent_of(root, leaf_node)
        while True:
            if p not in lineage_map:
                lineage_map[p] = i
                if p.edge_length == float("inf"):
                    break
                age += p.edge_length * tfac
                old_p = p
                p = parent_of(root, p)
                enopts.append((age, i, p.N / N0))
            else:
                # We have a join-on time
                ejopts.append((age, i, lineage_map[p]))
                break

    cmdline = ["-I %d %s" % (len(Iopts), " ".join(map(str, Iopts)))]
    for ej in ejopts:
        cmdline.append("-ej %g %d %d" % ej)
    for en in enopts:
        cmdline.append("-en %g %d %g" % en)
    cmdline = ["%s %d 1 -t %g" % (SCRM_PATH, sum(Iopts), theta)] + cmdline
    print(cmdline)
    return lineages, " ".join(cmdline)

def run_simulation(tree, L, lineages_per_taxon):
    lineages, cmd = build_command_line(tree, L, lineages_per_taxon)
    species = list(set(lineages))
    n_lineages = Counter(lineages)
    N0 = tree.N
    print(cmd)
    output = [l.strip() for l in check_output(cmd, shell=True).split("\n")]
    def f(x):
        if x == "//":
            f.i += 1
        return f.i
    f.i = 0 
    for k, lines in itertools.groupby(output, f):
        if k == 0:
            continue
        # Skip preamble
        next(lines)
        # segsites
        segsites = int(next(lines).split(" ")[1])
        # positions
        next(lines)
        # at haplotypes
        lin_counts = defaultdict(lambda: np.zeros(segsites, dtype=int))
        for hap, lin in zip(lines, lineages):
            hap = list(map(int, hap))
            lin_counts[lin] += hap
    return [{lin: NodeState(n_derived=lin_counts[lin][i], 
                            n_ancestral=n_lineages[lin] - lin_counts[lin][i]) 
                            for lin in lineages}
            for i in range(segsites)]

def build_splits(lineages, seqs_path, outgroup):
    splits = Counter()
    with open(seqs_path, "rt") as f:
        next(f)
        seqdata = [(lineages[int(spec) - 1], seq) for spec, seq in
                    (line.strip().split() for line in f)]
    specs = [s[0] for s in seqdata]
    c = Counter()
    nt = namedtuple("Spectrum", sorted({ell for ell in lineages if ell != outgroup}))
    for col in zip(*[s[1] for s in seqdata]):
        # This is not a true dict (potentially multiple of same key) but there should
        # be only one outgroup lineage
        dbase = dict(zip(specs, col))
        abase = dbase[outgroup]
        d = {}
        for spec, base in zip(specs, col):
            d.setdefault(spec, [0, 0])[int(base != abase)] += 1
        d = {k: tuple(v) for k, v in d.items() if k != outgroup}
        if not any(all(d[k][i] == 0 for k in specs if k != outgroup) for i in [0]):
            c[nt(**d)] += 1
    return c


if __name__ == "__main__":
    test_scrm_sim(mktree(10.0), "outgroup")
