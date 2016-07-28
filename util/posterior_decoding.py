#!/usr/bin/env python2.7

from __future__ import division
import numpy as np
import scipy.optimize
import scipy.ndimage
import pprint
import multiprocessing
import sys
import itertools
from collections import Counter
import sys
import argparse
import os, os.path

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 20})
import scrm 
import smcpp._smcpp, smcpp.util, smcpp.plotting, smcpp._newick, smcpp.estimation_tools

np.set_printoptions(linewidth=120, precision=6, suppress=True)

# Begin program
parser = argparse.ArgumentParser()
parser.add_argument("--N0", type=int, default=10000)
parser.add_argument("--theta", type=float, default=1.25e-8)
parser.add_argument("--rho", type=float, default=1.25e-8)
parser.add_argument("--estimated_rho", type=float, default=None)
parser.add_argument("--M", type=int, default=32, help="number of hidden states")
parser.add_argument("--panel-size", type=int, default=None)
parser.add_argument("--missing", type=float, default=None)
parser.add_argument("--error", type=float, default=None)
parser.add_argument("--thinning-factor", type=int, default=1)
parser.add_argument("--ascertain", type=float, metavar="x", default=0.0, help="delete all SNPs with MAF < x")
parser.add_argument("L", type=int)
parser.add_argument("seed", type=int, default=None)
# parser.add_argument("outpdf", type=str, help="Name of output pdf")
parser.add_argument("outdir", type=str, help="output directory")
parser.add_argument("demography", type=str)
parser.add_argument("n", type=int, nargs="+")
args = parser.parse_args()

# Make output directory
try:
    os.mkdir(args.outdir)
except OSError:
    pass

theta = args.theta
rho = args.rho
if args.estimated_rho is None:
    args.estimated_rho = rho

if args.demography == "human":
    demo = smcpp.util.human
    a = demo['a']
    b = demo['b']
    s = demo['s']
else:
    a, b, s = np.array(eval(args.demography))

true_parameters = (a, b, s)
width = 2000
G = np.zeros([args.M, args.L])
n_max = max(args.n)
if args.panel_size is None:
    args.panel_size = n_max
demography = scrm.demography_from_params((a, b, s * 0.5))
print(" ".join(map(str, demography)))
if args.seed is not None:
    np.random.seed(args.seed)
data0 = scrm.simulate(args.panel_size, args.N0, args.theta, args.rho, args.L, True, demography)

# Create missingness in data
if args.missing is not None:
    # randomly set some genotypes to missing
    dim = data0[2].shape
    nmiss = np.random.poisson(np.prod(dim) * args.missing)
    i1, i2 = [np.random.randint(0, upper, size=nmiss) for upper in dim]
    data0[2][i1, i2] = -1

# Draw from panel
data = smcpp.util.dataset_from_panel(data0, n_max, (0, 1), True)

# Dump trees
open(os.path.join(args.outdir, "trees.txt"), "wt").write("\n".join("[%d]%s" % c for c in data[3]))

# True coalescence times
ct = [(c1, 2. * smcpp._newick.tmrca(c2, "1", "2")) for c1, c2 in data[3]]
# Get rid of these it's really annoying when you accidentally print them.
data = data[:3]

if args.ascertain:
    maf = np.where(np.mean(data[2], axis=0) > args.ascertain)[0]
    data = (data[0], data[1][maf], data[2][:, maf])

# Inflate singletons 
# alpha fraction of called bases are false positive
if args.error is not None:
    # Each assayed position has an epsilon chance of being wrong.
    nerr = np.random.poisson(args.error * args.L * n_max)
    pos = zip(np.random.randint(0, n_max, nerr), np.random.randint(0, args.L, nerr))
    npos = np.random.random_integers(0, args.L - 1, size=nerr)
    ind = np.random.random_integers(0, n_max - 1, size=nerr)
    nhap = np.zeros([n_max, args.L], dtype=np.uint8)
    for i, j in enumerate(data[1]):
        nhap[:, j] = data[2][:, i]
    for i, j in zip(ind, npos):
        nhap[i, j] = 1 - nhap[i, j]
    npos = nhap.sum(axis=0) > 0
    data = (data[0], np.where(npos)[0], nhap[:, npos]) + data[3:]

gm = {}
full_gammas = {}
ims = {}
oo = {}
bks = {}
maps = {}
gammadict = {}
datadict = {}
for n in args.n:
    # subset data. some sites might be non-segregating in the subsample.
    seg = [i for i, pos in enumerate(data[1]) if 
            any(h[i] != 0 for h in data[2][:n])
            and not all(h[i] == 1 for h in data[2][:n])]
    segdata = np.array([[ba[i] for i in seg] for ba in data[2][:n]])
    dsub = (data[0], data[1][seg], segdata)
    obs = smcpp.util.hmm_data_format(dsub, n, (0, 1)).astype('int32')
    # oo[n] = np.array([c1[1:] for c1 in obs for _ in range(c1[0])])
    oo[n] = [smcpp.estimation_tools.thin_dataset([obs], 1)]
    oo[n], _ = smcpp.estimation_tools.break_long_spans(oo[n][0], np.inf, 0)
    hidden_states = smcpp._smcpp.balance_hidden_states((a,b,s), args.M + 1)
    im = smcpp._smcpp.PyInferenceManager(n - 2, oo[n], hidden_states)
    im.theta = 2.0 * args.N0 * args.theta
    im.rho = 2.0 * args.estimated_rho * args.N0
    im.save_gamma = True
    im.model = smcpp._smcpp.make_model([a, b, s])
    im.E_step()
    ims[n] = im
    # dump gamma file for each n
    gammadict[str(n)] = im.gammas[0]
    datadict[str(n)] = oo[n][0]
    gamma = np.zeros([args.M, args.L])
    sp = 0
    for row, col in zip(oo[n][0], im.gammas[0].T):
        an = row[0]
        gamma[:, sp:(sp+an)] = col[:, None] / an
        sp += an
    # gamma = gamma[:, 311000:313000]
    full_gammas[n] = gamma
    gm[n] = scipy.ndimage.zoom(gamma, (1.0, 1. * width / args.L))
    maps[n] = np.argmax(gamma, axis=0)

np.savetxt(os.path.join(args.outdir, "hidden_states.txt"), hidden_states)
np.savez_compressed(os.path.join(args.outdir, "posteriors.npz"), **gammadict)
np.savez_compressed(os.path.join(args.outdir, "data.npz"), **datadict)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=len(args.n), sharex=True, sharey=True, figsize=(25, 15))
if len(args.n) == 1:
    axes = [axes]
# for i, ll in enumerate(((0, 1), (1, 2))[:2]):
coal_times = np.searchsorted(hidden_states, list(smcpp.util.unpack(ct))) - 1
zct = scipy.ndimage.zoom(list(smcpp.util.unpack(ct)), 1. * width / args.L)
true_pos = scipy.ndimage.zoom(coal_times, 1. * width / args.L)
# axes[-1].step(range(width), true_pos)
#end i loop
plt.set_cmap("cubehelix")

ai = 0
label_text   = [r"%i kb" % int(args.L / 40. * 100. * loc/width) for loc in plt.xticks()[0]]
mx = max([np.max(gm[g]) for g in gm])
hs_mid = 0.5 * (hidden_states[1:] + hidden_states[:-1])
for n in sorted(gm):
    ax = axes[ai]
    ai += 1
    im = ax.imshow(gm[n][::-1], extent=[0, width, -0.5, args.M - 0.5],aspect='auto', vmin=0.0, vmax=mx)
    map_pos = scipy.ndimage.zoom(maps[n], 1. * width / args.L)
    ax.step(range(width), map_pos, color="red")
    ax.step(range(width), true_pos, color=(0., 1., 0.))
    diff = np.abs(np.subtract.outer(hs_mid, zct))
    # escore = (diff * gm[n]).sum() / diff.shape[1]
    ax.set_ylabel("n=%d" % n)
    ax.set_ylim([-0.5, args.M - 0.5])
    ax.set_xticklabels(label_text)
    corr = np.corrcoef(coal_times, maps[n])[0, 1]
    txt = ax.text(width + 35, 5, "%.4f" % corr, rotation=-90, va='bottom', ha='right')
smcpp.plotting.save_pdf(fig, os.path.join(args.outdir, "plot.pdf"))
plt.close(fig)
