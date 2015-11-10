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
import psmcpp, psmcpp.scrm, psmcpp.util

parser = argparse.ArgumentParser()
parser.add_argument("seed", type=int)
parser.add_argument("outdir")
parser.add_argument("-a", type=float, nargs="+", required=True)
parser.add_argument("-b", type=float, nargs="+", required=True)
parser.add_argument("-s", type=float, nargs="+", required=True)
parser.add_argument("n", type=int, help="Sample size.")
parser.add_argument("L", type=int, help="Length of each simulated chromosome.")
parser.add_argument("C", type=int, help="Number of chromosomes to simulate.")
parser.add_argument("--panel_size", type=int, default=None, help="Panel size (SMC++ only)")
parser.add_argument("--N0", type=int, default=10000)
parser.add_argument("--theta", type=float, default=1.25e-8, help="Unscaled mutation rate")
parser.add_argument("--rho", type=float, default=4 * 1.25e-8, help="Unscaled recombination rate")
parser.add_argument("--switch_error", type=float, default=0.02, 
        help="Probability of switch error for simulating phasing.")
args = parser.parse_args()

def mk_outdir(prog):
    ret = os.path.join(args.outdir, prog)
    try:
        os.makedirs(ret)
    except OSError:
        pass
    return ret

np.random.seed = args.seed
a0 = np.array(args.a)
b0 = np.array(args.b)
s0 = np.array(args.s)
if args.panel_size is None:
    args.panel_size = args.n

demography = psmcpp.scrm.demography_from_params((a0 * 2.0, b0 * 2.0, s0))

# Generate data set using scrm
print("simulating")
def perform_sim(args):
    n, N0, theta, rho, L, demography, seed = args
    return psmcpp.scrm.simulate(n, N0, theta, rho, L, demography, False, seed)

p = multiprocessing.Pool(16)
data_sets = list(p.imap_unordered(perform_sim, 
    [(args.panel_size, args.N0, args.theta, args.rho, args.L, demography, np.random.randint(0, sys.maxint)) 
        for _ in range(args.C)]))
p.terminate()
p.join()
del p

# 1. Write smc++ format
smcpp_outdir = mk_outdir("smc++")
obs = [psmcpp.util.hmm_data_format(data, args.n, (0, 1)) for data in data_sets]
with open(os.path.join(smcpp_outdir, "smc++.dat"), "wb") as f:
    pickle.dump(f, obs)

# 2. Write psmc format
# psmcfa format: N: missing, K: het, T: homo, block_len: 100, linewidth 60
# with open(os.path.join(args.outdir, "psmc.psmcfa"), "wt") as f:
#     for i, obs in enumerate(obs_list, 1):
#         f.write(">seq{i}\n".format(i=i))
#         obsiter = ((x[0], x[1:]) for x in obs)
#         seqs = ["K" if np.sum(ob, axis=0)[0] > 0 else "T" for ob in psmcpp.util.grouper(psmcpp.util.unpack(obsiter), 100)]
#         f.writelines("".join(sq) + "\n" for sq in psmcpp.util.grouper(seqs, 60, ""))


# The next two methods require phased data. To model the effect of
# computational phasing algorithms we create haplotypes with switch
# error.

# 3. Write msmc / dical format
msmc_template = "seq{chrom}\t{pos}\t{dist}\t{gts}\n"
msmc_bps = np.array(["A", "G"])
msmc_outdir = os.path.join(args.outdir, "msmc")
try:
    os.makedirs(msmc_outdir)
except OSError:
    pass
# Dical
dical_dir = os.path.join(args.outdir, "dical")
try:
    os.makedirs(dical_dir)
except OSError:
    pass
with open(os.path.join(dical_dir, "dical.vcf"), "wt") as dical_vcf:
    dical_vcf.write("#")
    dical_vcf.write("\t".join("CHROM POS ID REF ALT QUAL FILTER INFO".split()))
    dical_vcf.write("\t")
    dical_vcf.write("\t".join(["IND%i" % k for k in range(1, args.n // 2 + 1)]))
    dical_vcf.write("\n")
    for c, data_set in enumerate(data_sets, 1):
        with open(os.path.join(msmc_outdir, "seq{chrom}.txt".format(chrom=c)), "wt") as f:
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
                dical_vcf.write("\t".join(rec) + "\n")
                bps = msmc_bps[gts]
                f.write(msmc_template.format(chrom=c, pos=pos, dist=dist, gts="".join(bps)))
