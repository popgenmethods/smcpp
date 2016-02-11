#!/usr/bin/env python2.7
'''Generate simulated data sets for the various inference packages.'''

import numpy as np
import random
import sys
sys.path.append("util/")
import multiprocessing
import argparse
import cPickle as pickle
import os.path
import vcf
import cStringIO
from collections import Counter

import scrm, smcpp.util

def perform_sim(args):
    n, N0, theta, rho, L, demography, seed, trees, missing = args
    np.random.seed(seed)
    ret = scrm.simulate(n, N0, theta, rho, L, demography, trees)
    positions, haps = ret[1:3]
    c = Counter(haps.sum(axis=0))
    psfs = [0] * (n - 1)
    for s in c:
        assert 0 < s < n
        psfs[s - 1] = c[s]
    print(psfs)
    if missing > 0:
        new_positions = []
        new_haps = []
        pos = 0
        pos_i = 0
        M = haps.shape[0]
        rate = 1. - (1. - missing)**M
        while True:
            pos = min(positions[pos_i], pos + np.random.geometric(rate))
            if pos > L:
                break
            nb = np.random.binomial(M, missing)
            while nb == 0:
                nb = np.random.binomial(M, missing)
            if pos == positions[pos_i]:
                new_hap = haps[:, pos_i]
                pos_i += 1
            else:
                new_hap = np.zeros(M, dtyp=np.int32)
            new_hap[np.random.sample(nb)] = -1
            new_positions.append(pos)
            new_haps.append(new_hap)
        ret = (L, new_positions, np.array(new_haps, dtype=np.int32).T) + ret[3:]
    return ret

def process_psmc(args):
    data, n = args
    obs = smcpp.util.hmm_data_format(data, n, (0, 1))
    obsiter = ((x[0], x[1:]) for x in obs)
    seqs = ("K" if np.sum(ob, axis=0)[0] > 0 else "T" for ob in smcpp.util.grouper(smcpp.util.unpack(obsiter), 100))
    return "\n".join("".join(sq) for sq in smcpp.util.grouper(seqs, 60, ""))

def process_msmc(tup):
    c, data_set, msmc_template, msmc_outdir, args = tup
    msmc_txt = open(os.path.join(msmc_outdir, "seq{chrom}.txt".format(chrom=c)), "wt")
    lpos = -1
    # Assume phasing is correct initially
    phase_error = np.zeros(args.msmc_sample_size, dtype=bool)
    for i, pos in enumerate(data_set[1]):
        rec = ["seq%i" % c, str(pos), ".", ".", ".", "70", "PASS", "."]
        if data_set[2][:args.msmc_sample_size, i].sum() == 0:
            continue
        dist = pos - lpos
        lpos = pos
        gts = list(data_set[2][:args.msmc_sample_size, i])
        for i in range(len(gts) // 2):
            phase_error[i] = phase_error[i] ^ (np.random.random() < args.switch_error)
            gts[i], gts[i + 1] = gts[i + phase_error[i]], gts[i + 1 - phase_error[i]]
            rec.append("%i|%i" % (gts[i], gts[i + 1]))
        msmc_txt.write(msmc_template.format(chrom=c, pos=pos, dist=dist, gts="".join(map(str, gts))))

def process_tmrca(args):
    fn, trees = args
    A = []
    from smcpp._newick import tmrca
    last_d12 = None
    sp = 0
    for span, tree in trees:
        d12 = tmrca(tree, "1", "2")
        if last_d12 is None or d12 == last_d12:
            sp += span
        else:
            A.append((sp, last_d12))
            sp = span
        last_d12 = d12
    A.append((sp, last_d12))
    np.savetxt(fn, A, fmt="%g")

def savetxt(args):
    np.savetxt(*args, fmt="%i")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", type=int)
    parser.add_argument("outdir")
    parser.add_argument("--a", type=float, nargs="+")
    parser.add_argument("--b", type=float, nargs="+")
    parser.add_argument("--s", type=float, nargs="+")
    parser.add_argument("n", type=int, help="Sample size.")
    parser.add_argument("L", type=int, help="Length of each simulated chromosome.")
    parser.add_argument("C", type=int, help="Number of chromosomes to simulate.")
    parser.add_argument("--pairs", type=int, default=1, help="Number of pairs to generate")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--sawtooth", const=True, default=False, action="store_const")
    group.add_argument("--human", const=True, default=False, action="store_const")
    parser.add_argument("--panel-size", type=int, default=None, help="Panel size (SMC++ only)")
    parser.add_argument("--N0", type=int, default=10000)
    parser.add_argument("--theta", type=float, default=1.25e-8, help="Unscaled mutation rate")
    parser.add_argument("--trees", default=False, action="store_true")
    parser.add_argument("--rho", type=float, default=1.25e-8, help="Unscaled recombination rate")
    parser.add_argument("--missing", type=float, default=0., help="Missingness fraction for data")
    parser.add_argument("--switch-error", type=float, default=0.02, 
            help="Probability of switch error for simulating phasing.")
    parser.add_argument("--smcpp", action="store_true", default=True, help="Generate dataset for SMC++")
    parser.add_argument("--psmc", action="store_true", default=False, help="Generate dataset for PSMC")
    parser.add_argument("--msmc", action="store_true", default=False, help="Generate dataset for MSMC")
    parser.add_argument("--msmc-sample-size", default=None, type=int, help="Use subset of size <k> for MSMC dataset")
    # parser.add_argument("--dical", action="store_true",  default=False, help="Generate dataset for diCal")
    args = parser.parse_args()
    assert 0. <= args.missing < 1

    def mk_outdir(prog):
        ret = os.path.join(args.outdir, prog)
        try:
            os.makedirs(ret)
        except OSError:
            pass
        return ret

    np.random.seed(args.seed)

    if args.sawtooth:
        st = smcpp.util.sawtooth
        a0 = st['a']
        b0 = st['b']
        s0 = st['s']
    elif args.human:
        hum = smcpp.util.human
        a0 = hum['a'] / (2. * args.N0)
        b0 = hum['b'] / (2. * args.N0)
        s0 = hum['s_gen'] / (2. * args.N0)
    else:
        a0 = np.array(args.a)
        b0 = np.array(args.b)
        s0 = np.array(args.s)

    if args.panel_size is None:
        args.panel_size = 2 * args.n

    demography = scrm.demography_from_params((a0, b0, s0 * 0.5))

# Generate data set using scrm
    print("simulating")

    pool = multiprocessing.Pool(args.C)
    data_sets = list(pool.imap_unordered(perform_sim, 
        [(args.panel_size, args.N0, args.theta, args.rho, args.L, 
            demography, np.random.randint(0, 4000000000), args.trees,
            args.missing) for _ in range(args.C)]))
    assert data_sets

    try:
        os.makedirs(args.outdir)
    except OSError:
        pass
    open(os.path.join(args.outdir, "meta.txt"), "wt").write(
            "{argv0} created this dataset. The command line was:\n\t{cmd_line}\nThe args object looks like:\n{args}\nThe ms demography was\n\t{demo}\n".format(
            argv0=sys.argv[0], args=args, cmd_line=" ".join(sys.argv), demo=demography))

    assert args.pairs <= args.n
    pairs = [(2 * k, 2 * k + 1) for k in range(args.pairs)]

    if args.trees:
        tree_dir = mk_outdir("trees")
        pool.map(process_tmrca, [(os.path.join(tree_dir, "%i.txt.gz") % i, data[3]) for i, data in enumerate(data_sets)])

# 1. Write smc++ format
    if args.smcpp:
        smcpp_outdir = mk_outdir("smc++")
        obs = [smcpp.util.hmm_data_format(data, 2 * args.n, p) for data in data_sets for p in pairs]
        pool.map(savetxt, [(os.path.join(smcpp_outdir, "%i.txt.gz" % i), ob) for i, ob in enumerate(obs, 1)])

    if not(any([args.psmc, args.msmc])): sys.exit(0)

# 2. Write psmc format
    if args.psmc:
# psmcfa format: N: missing, K: het, T: homo, block_len: 100, linewidth 60
        psmc_outdir = mk_outdir("psmc")
        with open(os.path.join(psmc_outdir, "psmc.psmcfa"), "wt") as f:
            for i, seq in enumerate(pool.map(process_psmc, [(data, args.n) for data in data_sets])):
                f.write(">seq{i}\n{seq}\n".format(i=i, seq=seq))

# The next two methods require phased data. To model the effect of
# computational phasing algorithms we create haplotypes with switch
# error.

# 3. Write msmc / dical format
    if args.msmc:
        if args.msmc_sample_size is None:
            args.msmc_sample_size = 2 * args.n
        msmc_template = "seq{chrom}\t{pos}\t{dist}\t{gts}\n"
        msmc_outdir = os.path.join(args.outdir, "msmc")
        try:
            os.makedirs(msmc_outdir)
        except OSError:
            pass
    pool.map(process_msmc, [(i, ds, msmc_template, msmc_outdir, args) for i, ds in enumerate(data_sets)])
