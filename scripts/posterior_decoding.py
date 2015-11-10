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
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 32})
import psmcpp.scrm, psmcpp._pypsmcpp, psmcpp.util, psmcpp.plotting, psmcpp._newick
np.set_printoptions(linewidth=120, precision=6, suppress=True)

# Begin program
parser = argparse.ArgumentParser()
parser.add_argument("--block_size", type=int, default=50)
parser.add_argument("--N0", type=int, default=10000)
parser.add_argument("--theta", type=float, default=1.25e-8)
parser.add_argument("--rho", type=float, default=1.25e-8 / 4.0)
parser.add_argument("--M", type=int, default=32, help="number of hidden states")
parser.add_argument("--panel-size", type=int, default=None)
parser.add_argument("L", type=int)
parser.add_argument("seed", type=int)
parser.add_argument("outpdf", type=str, help="Name of output pdf")
parser.add_argument("-a", type=float, nargs="+")
parser.add_argument("-b", type=float, nargs="+")
parser.add_argument("-s", type=float, nargs="+")
parser.add_argument("-ns", type=int, nargs="+")
args = parser.parse_args()

theta = 1.25e-8
rho = theta / 4.0
a = np.array(args.a)
b = np.array(args.b)
s = np.array(args.s)
true_parameters = (a, b, s)
width = 2000
G = np.zeros([args.M, args.L])
n = max(args.ns)
if args.panel_size is None:
    args.panel_size = n
demography = psmcpp.scrm.demography_from_params((a * 2.0, b * 2.0, s))
print(" ".join(map(str, demography)))
data = psmcpp.scrm.simulate(args.panel_size, args.N0, args.theta, args.rho, 
        args.L, demography, include_trees=True, seed=args.seed)
data = psmcpp.util.dataset_from_panel(data, n, (0, 1))
# True coalescence times
ct = [(c1, psmcpp._newick.tmrca(c2, "1", "2")) for c1, c2 in data[3]]

# Inflate singletons 
# alpha fraction of called bases are false positive
if False:
    alpha = .001
    err_bases = int(L * n * alpha)
    npos = np.random.random_integers(0, L - 1, size=err_bases)
    ind = np.random.random_integers(0, n - 1, size=err_bases)
    nhap = np.zeros([n, L], dtype=np.uint8)
    for i, j in enumerate(data[1]):
        nhap[:, j] = data[2][:, i]
    for i, j in zip(ind, npos):
        nhap[i, j] = 1 - nhap[i, j]
    npos = nhap.sum(axis=0) > 0
    data = (data[0], np.where(npos)[0], nhap[:, npos], data[3])


gm = {}
ims = {}
oo = {}
bks = {}
for n in args.ns:
    # subset data. some sites might be non-segregating in the subsample.
    seg = [i for i, pos in enumerate(data[1]) if any(not h[i] for h in data[2][:n])]
    segdata = np.array([[ba[i] for i in seg] for ba in data[2][:n]])
    dsub = (data[0], data[1][seg], segdata, data[3])
    obs = psmcpp.util.hmm_data_format(dsub, n, (0, 1))
    oo[n] = np.array([c1[1:] for c1 in obs for _ in range(c1[0])])
    hidden_states = np.array([0., 14.0])
    im = psmcpp._pypsmcpp.PyInferenceManager(n - 2, [obs[:10]], hidden_states,
            4.0 * args.N0 * args.theta, 4.0 * args.N0 * args.rho, args.block_size, 10, [0])
    hidden_states = im.balance_hidden_states((a, b, s), args.M)
    hidden_states[-1] = 14.9
    em = np.arange(3 *  (n - 1), dtype=int).reshape([3, n - 1])
    em[0] = em[2] = 0
    em[1] = 1
    im = psmcpp._pypsmcpp.PyInferenceManager(n - 2, [obs], hidden_states,
            4.0 * args.N0 * args.theta, 4.0 * args.N0 * args.rho, args.block_size, 1, [0], em)
    im.setParams((a, b, s), False)
    im.Estep()
    ims[n] = im
    gamma = im.gammas()[0]
    bks[n] = im.block_keys()[0]
    bb = 0
    for i, (_, d) in enumerate(bks[n]):
        w = sum(d.values())
        G[:,bb:(bb+w)] = gamma[:,i:(i+1)]
        bb += w
    gm[n] = scipy.ndimage.zoom(G, (1.0, 1. * width / args.L))

import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=len(args.ns), sharex=True, sharey=True, figsize=(25, 15))
# for i, ll in enumerate(((0, 1), (1, 2))[:2]):
coal_times = np.searchsorted(hidden_states, list(psmcpp.util.unpack(ct))) - 1
true_pos = scipy.ndimage.zoom(coal_times, 1. * width / args.L)
ai = 0
label_text   = [r"%i kb" % int(args.L / 40. * 100. * loc/width) for loc in plt.xticks()[0]]
mx = max([np.max(gm[g]) for g in gm])
for n in sorted(gm):
    ax = axes[ai]
    ai += 1
    im = ax.imshow(gm[n][::-1], extent=[0, width, -0.5, args.M - 0.5],aspect='auto', vmin=0.0, vmax=mx)
    ax.step(range(width), true_pos, color=(0, 1., 1.))
    ax.set_ylabel("n=%d" % n)
    ax.set_ylim([-0.5, args.M - 0.5])
    ax.set_xticklabels(label_text)
# axes[-1].set_ylabel("True hid. st.")
# axes[-1].set_ylim([-0.5, M - 0.5])
# axes[-1].xaxis.set_ticks(np.arange(0, width + 1, 100))
# fig.subplots_adjust(right=0.9)
# cbar_ax = fig.add_axes([0.92, 0.29, 0.02, 0.55])
# fig.colorbar(im, cax=cbar_ax)
psmcpp.plotting.save_pdf(fig, args.outpdf)
plt.close(fig)
