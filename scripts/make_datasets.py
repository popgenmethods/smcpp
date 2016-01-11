#!/usr/bin/env python2.7
'''Generated simulated data sets for the various inference packages.'''

import numpy as np
import random
import sys
import multiprocessing
import argparse
import cPickle as pickle
import os.path
import vcf
import cStringIO
import psmcpp, psmcpp.lib.scrm, psmcpp.lib.util

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
parser.add_argument("--sawtooth", const=True, default=False, action="store_const")
parser.add_argument("--panel-size", type=int, default=None, help="Panel size (SMC++ only)")
parser.add_argument("--N0", type=int, default=10000)
parser.add_argument("--theta", type=float, default=1.25e-8, help="Unscaled mutation rate")
parser.add_argument("--rho", type=float, default=1.25e-8 / 4.0, help="Unscaled recombination rate")
parser.add_argument("--missing", type=float, default=0., help="Missingness fraction for data")
parser.add_argument("--switch-error", type=float, default=0.02, 
        help="Probability of switch error for simulating phasing.")
parser.add_argument("--smcpp", action="store_true", default=True, help="Generate dataset for SMC++")
parser.add_argument("--psmc", action="store_true", default=False, help="Generate dataset for PSMC")
parser.add_argument("--msmc", action="store_true", default=False, help="Generate dataset for MSMC")
parser.add_argument("--dical", action="store_true",  default=False, help="Generate dataset for diCal")
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
    st = psmcpp.lib.util.sawtooth
    a0 = st['a']
    b0 = st['b']
    s0 = st['s_gen'] / (2. * args.N0)
# MSMC sample demography
else:
    a0 = np.array(args.a)
    b0 = np.array(args.b)
    s0 = np.array(args.s)

if args.panel_size is None:
    args.panel_size = 2 * args.n

demography = psmcpp.lib.scrm.demography_from_params((a0 * 2.0, b0 * 2.0, s0))

# Generate data set using scrm
print("simulating")
def perform_sim(args):
    n, N0, theta, rho, L, demography, seed = args
    np.random.seed(seed)
    return psmcpp.lib.scrm.simulate(n, N0, theta, rho, L, demography, False)

p = multiprocessing.Pool(args.C)
data_sets = list(p.imap_unordered(perform_sim, 
    [(args.panel_size, args.N0, args.theta, args.rho, args.L, demography, np.random.randint(0, 4000000000)) for _ in range(args.C)]))
assert data_sets
p.terminate()
p.join()
del p

try:
    os.makedirs(args.outdir)
except OSError:
    pass
open(os.path.join(args.outdir, "meta.txt"), "wt").write(
        "{argv0} created this dataset. The command line was:\n\t{cmd_line}\nThe args object looks like:\n{args}".format(
        argv0=sys.argv[0], args=args, cmd_line=" ".join(sys.argv)))

assert args.pairs <= args.n
pairs = [(2 * k, 2 * k + 1) for k in range(args.pairs)]

# 1. Write smc++ format
if args.smcpp:
    smcpp_outdir = mk_outdir("smc++")
    obs = [psmcpp.lib.util.hmm_data_format(data, 2 * args.n, p, missing=args.missing) for data in data_sets for p in pairs]
    for i, ob in enumerate(obs, 1):
        np.savetxt(os.path.join(smcpp_outdir, "%i.txt.gz" % i), ob, fmt="%i")

if not(any([args.psmc, args.dical, args.msmc])): sys.exit(0)

# 2. Write psmc format
if args.psmc:
# psmcfa format: N: missing, K: het, T: homo, block_len: 100, linewidth 60
    psmc_outdir = mk_outdir("psmc")
    with open(os.path.join(psmc_outdir, "psmc.psmcfa"), "wt") as f:
        for i, data in enumerate(data_sets, 1):
            obs = psmcpp.lib.util.hmm_data_format(data, args.n, (0, 1))
            f.write(">seq{i}\n".format(i=i))
            obsiter = ((x[0], x[1:]) for x in obs)
            seqs = ["K" if np.sum(ob, axis=0)[0] > 0 else "T" for ob in psmcpp.lib.util.grouper(psmcpp.lib.util.unpack(obsiter), 100)]
            f.writelines("".join(sq) + "\n" for sq in psmcpp.lib.util.grouper(seqs, 60, ""))


# The next two methods require phased data. To model the effect of
# computational phasing algorithms we create haplotypes with switch
# error.

# 3. Write msmc / dical format
if args.msmc:
    msmc_template = "seq{chrom}\t{pos}\t{dist}\t{gts}\n"
    msmc_bps = np.array(["A", "G"])
    msmc_outdir = os.path.join(args.outdir, "msmc")
    try:
        os.makedirs(msmc_outdir)
    except OSError:
        pass
# Dical
if args.dical:
    dical_dir = os.path.join(args.outdir, "dical")
    try:
        os.makedirs(dical_dir)
    except OSError:
        pass
    dical_vcf = open(os.path.join(dical_dir, "dical.vcf"), "wt")
    dical_vcf.write("#")
    dical_vcf.write("\t".join("CHROM POS ID REF ALT QUAL FILTER INFO".split()))
    dical_vcf.write("\t")
    dical_vcf.write("\t".join(["IND%i" % k for k in range(1, args.n // 2 + 1)]))
    dical_vcf.write("\n")

for c, data_set in enumerate(data_sets, 1):
    if args.msmc:
        msmc_txt = open(os.path.join(msmc_outdir, "seq{chrom}.txt".format(chrom=c)), "wt")
    lpos = 0
    phase = np.zeros(args.n, dtype=bool)
    for i, pos in enumerate(data_set[1]):
        rec = ["seq%i" % c, str(pos), ".", ".", ".", "70", "PASS", "."]
        dist = pos - lpos
        lpos = pos
        gts = list(data_set[2][:, i])
        for i in range(len(gts) // 2):
            if gts[i] + gts[i + 1] > 0:
                phase[i] != (random.random() < args.switch_error)
            gts[i], gts[i + 1] = gts[i + phase[i]], gts[i + 1 - phase[i]]
            rec.append("%i|%i" % (gts[i], gts[i + 1]))
        if args.dical:
            dical_vcf.write("\t".join(rec) + "\n")
        bps = msmc_bps[gts]
        if args.msmc:
            msmc_txt.write(msmc_template.format(chrom=c, pos=pos, dist=dist, gts="".join(bps)))
