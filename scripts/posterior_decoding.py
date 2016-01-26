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
import psmcpp.lib.scrm, psmcpp._pypsmcpp, psmcpp.lib.util, psmcpp.lib.plotting, psmcpp._newick
np.set_printoptions(linewidth=120, precision=6, suppress=True)

# Begin program
parser = argparse.ArgumentParser()
parser.add_argument("--N0", type=int, default=10000)
parser.add_argument("--theta", type=float, default=1.25e-8)
parser.add_argument("--rho", type=float, default=1.25e-8)
parser.add_argument("--M", type=int, default=32, help="number of hidden states")
parser.add_argument("--panel-size", type=int, default=None)
parser.add_argument("--alt-freq", type=int, default=None)
parser.add_argument("--missing", type=float, default=None)
parser.add_argument("L", type=int)
parser.add_argument("seed", type=int, default=None)
parser.add_argument("outpdf", type=str, help="Name of output pdf")
parser.add_argument("-a", type=float, nargs="+")
parser.add_argument("-b", type=float, nargs="+")
parser.add_argument("-s", type=float, nargs="+")
parser.add_argument("-ns", type=int, nargs="+")
args = parser.parse_args()

hidden_states = np.array([
0.,0.022651,0.029053,0.037265,0.037273,0.044289,0.047798,0.052648,0.061308,0.063315,
0.069434,0.078637,0.082681,0.089059,0.100863,0.105106,0.114232,0.129372,0.133422,0.146519,
0.165939,0.17141,0.187933,0.212842,0.223077,0.241052,0.273002,0.292716,0.309186,0.350165,
0.383141,0.396577,0.44914,0.495336,0.508669,0.576089,0.627677,0.652444,0.73892,0.776481,
0.836858,0.940592,0.947776,1.073395,1.122644,1.215665,1.326254,1.37679,1.555871,1.559272,1.76594,
1.82735,2.,2.153478,2.507998,2.896332,3.325614,3.805511,4.349574,4.977648,5.720502,
6.629682,7.801818,9.453852,12.278022,49.999])
args.M = len(hidden_states) - 1

theta = 1.25e-8
rho = theta / 4
demo = psmcpp.lib.util.human
a = demo['a']
b = demo['b']
s = demo['s_gen'] / (2. * args.N0)
true_parameters = (a, b, s)
width = 2000
G = np.zeros([args.M, args.L])
n_max = max(args.ns)
if args.panel_size is None:
    args.panel_size = n_max
demography = psmcpp.lib.scrm.demography_from_params((a, b, s * 0.5))
print(" ".join(map(str, demography)))
if args.seed is not None:
    np.random.seed(args.seed)
data0 = psmcpp.lib.scrm.simulate(args.panel_size, args.N0, args.theta, args.rho, 
        args.L, demography, include_trees=True)

# Create missingness in data
if args.missing is not None:
    # randomly set some genotypes to missing
    inds = np.random.random(data0[2].shape) < args.missing
    data0[2][inds] = -1

# Draw from panel
data = psmcpp.lib.util.dataset_from_panel(data0, n_max, (0, 1), True)

# True coalescence times
ct = [(c1, 2. * psmcpp._newick.tmrca(c2, "1", "2")) for c1, c2 in data[3]]
# Get rid of these it's really annoying when you accidentally print them.
data = data[:3]

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
    data = (data[1], np.where(npos)[0], nhap[:, npos], data[3])

gm = {}
ims = {}
oo = {}
bks = {}
for n in args.ns:
    # subset data. some sites might be non-segregating in the subsample.
    seg = [i for i, pos in enumerate(data[1]) if 
            any(h[i] != 0 for h in data[2][:n])
            and not all(h[i] == 1 for h in data[2][:n])]
    segdata = np.array([[ba[i] for i in seg] for ba in data[2][:n]])
    dsub = (data[0], data[1][seg], segdata)
    obs = psmcpp.lib.util.hmm_data_format(dsub, n, (0, 1))
    # oo[n] = np.array([c1[1:] for c1 in obs for _ in range(c1[0])])
    oo[n] = psmcpp.lib.util.normalize_dataset(obs, n)
    # hidden_states = psmcpp._pypsmcpp.balance_hidden_states((a,b,s), args.M)
    # hidden_states[0] = 0.
    # hidden_states[-1] = 49.9
    im = psmcpp._pypsmcpp.PyInferenceManager(n - 2, oo[n], hidden_states,
            2.0 * args.N0 * args.theta, 2.0 * args.N0 * args.rho)
    im.saveGamma = True
    im.setParams((a, b, s), False)
    im.Estep()
    ims[n] = im
    gamma = np.zeros([args.M, args.L])
    sp = 0
    for row, col in zip(oo[n][0], im.gammas[0].T):
        an = row[0]
        gamma[:, sp:(sp+an)] = col[:, None] / an
        sp += an
    # gamma = gamma[:, 311000:313000]
    gm[n] = scipy.ndimage.zoom(gamma, (1.0, 1. * width / args.L))

import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=len(args.ns), sharex=True, sharey=True, figsize=(25, 15))
# for i, ll in enumerate(((0, 1), (1, 2))[:2]):
coal_times = np.searchsorted(hidden_states, list(psmcpp.lib.util.unpack(ct))) - 1
zct = scipy.ndimage.zoom(list(psmcpp.lib.util.unpack(ct)), 1. * width / args.L)
true_pos = scipy.ndimage.zoom(coal_times, 1. * width / args.L)
# axes[-1].step(range(width), true_pos)
#end i loop
#plt.set_cmap("cubehelix")

ai = 0
label_text   = [r"%i kb" % int(args.L / 40. * 100. * loc/width) for loc in plt.xticks()[0]]
mx = max([np.max(gm[g]) for g in gm])
hs_mid = 0.5 * (hidden_states[1:] + hidden_states[:-1])
for n in sorted(gm):
    ax = axes[ai]
    ai += 1
    im = ax.imshow(gm[n][::-1], extent=[0, width, -0.5, args.M - 0.5],aspect='auto', vmin=0.0, vmax=mx)
    ax.step(range(width), true_pos, color=(0, 1., 1.))
    diff = np.abs(np.subtract.outer(hs_mid, zct))
    # escore = (diff * gm[n]).sum() / diff.shape[1]
    ax.set_ylabel("n=%d" % n)
    ax.set_ylim([-0.5, args.M - 0.5])
    ax.set_xticklabels(label_text)
    # txt = ax.text(width + 35, 0, "%.4f" % escore, rotation=-90, va='bottom', ha='right')
psmcpp.lib.plotting.save_pdf(fig, args.outpdf)
plt.close(fig)
