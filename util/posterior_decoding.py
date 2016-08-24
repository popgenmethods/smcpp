#!/usr/bin/env python2.7

from __future__ import division, print_function
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
import logging

from stepfun import StepFunction

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 20})
from matplotlib.image import NonUniformImage

import scrm 
import smcpp._smcpp, smcpp.util, smcpp.plotting, smcpp._newick, smcpp.estimation_tools, smcpp.model
from smcpp.model import PiecewiseModel

np.set_printoptions(linewidth=120, precision=6, suppress=True)

# Begin program
parser = argparse.ArgumentParser()
parser.add_argument("--plot", default=False, action="store_true")
parser.add_argument("--N0", type=int, default=10000)
parser.add_argument("--theta", type=float, default=1.25e-8)
parser.add_argument("--rho", type=float, default=1.25e-8)
parser.add_argument("--estimated-rho", type=float)
parser.add_argument("--M", type=int, default=32, help="number of hidden states")
parser.add_argument("--panel-size", type=int)
parser.add_argument("--missing", type=float)
parser.add_argument("--error", type=float)
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
    # a = np.array([1.0, 1.0])
    # b = np.array([1.0, 1.0])
    # s = np.array([1.0, 1.0])
    # a = np.array([3., 2., 1.])
    # b = np.array([3., 2., 1.])
    # s = np.array([0.01, 0.01, 0.01])
    model = smcpp.model.OldStyleModel(a, b, s)
    print(np.array([model.s, model.stepwise_values()]).T)
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
demography += ['-l', 0]
data0 = scrm.simulate(args.panel_size, args.N0, args.theta, args.rho, args.L, True, demography)
print("done simulating")

def perturb_tract(data, rate, f, ub=2, mtl=1000):
    mtl = 1000  # mean_tract_length
    num_tracts = np.random.poisson(rate * args.L / mtl) # mu * mtl / L = rate
    tract_len = np.random.geometric(1. / mtl, size=num_tracts)
    starts = np.random.randint(0, args.L, size=num_tracts)
    # nerr = np.random.poisson(error * args.L * ub)
    print("errors: %d" % tract_len.sum())
    # npos = np.random.randint(0, args.L, size=nerr)
    # ind = np.random.randint(0, ub, size=nerr)
    nhap = np.zeros([n_max, args.L], dtype=np.int8)
    nhap[:, data[1]] = data[2]
    for st, tr in zip(starts, tract_len):
        print(st, tr)
        nhap[:2, st:st + tr] = -1
    # nhap[ind, npos] = f(nhap[ind, npos])
    npos = np.any(nhap != 0, axis=0)
    return (data[0], np.where(npos)[0], nhap[:, npos]) + data[3:]

def perturb(data, error, f):
    nerr = np.random.poisson(error * args.L * n_max)
    print("errors: %d" % nerr)
    npos = np.random.randint(0, args.L, size=nerr)
    ind = np.random.randint(0, n_max, size=nerr)
    nhap = np.zeros([n_max, args.L], dtype=np.int8)
    nhap[:, data[1]] = data[2]
    nhap[ind, npos] = f(nhap[ind, npos])
    npos = nhap.sum(axis=0) > 0
    return (data[0], np.where(npos)[0], nhap[:, npos]) + data[3:]

def perturb_simple(data, error, f):
    nerr = np.random.poisson(error * np.prod(ary.shape))
    print("errors: %d" % nerr)
    pos = np.random.randint(0, ary.shape[0], size=nerr)
    ind = np.random.randint(0, ary.shape[1], size=nerr)
    ary[pos, ind] = f(ary[pos, ind])
    return data

if args.error:
    data0 = perturb(data0, args.error, lambda x: 1 - x)

if args.missing:
    data0 = perturb(data0, args.missing, lambda x: -1)

# Draw from panel
data = smcpp.util.dataset_from_panel(data0, n_max, (0, 1), True)
print("done panel")

# Dump trees
open(os.path.join(args.outdir, "trees.txt"), "wt").write("\n".join("[%d]%s" % c for c in data[3]))

# True coalescence times
ct = [(c1, 2 * smcpp._newick.tmrca(c2, "1", "2")) for c1, c2 in data[3]]
tct = np.array(list(smcpp.util.unpack(ct)))
pct = np.linspace(0, 100, args.M, False)[1:]
uct = list({x for _, x in ct})
hidden_states = np.percentile(tct, pct)
hidden_states = np.concatenate([[0.], hidden_states, [np.inf]])
print(hidden_states)
hidden_states = smcpp.estimation_tools.balance_hidden_states(model, args.M + 1)
print(hidden_states)
# Get rid of these it's really annoying when you accidentally print them.
data = data[:3]

if args.ascertain:
    maf = np.where(np.mean(data[2], axis=0) > args.ascertain)[0]
    data = (data[0], data[1][maf], data[2][:, maf])

gm = {}
full_gammas = {}
ims = {}
oo = {}
bks = {}
maps = {}
gammadict = {}
datadict = {}
for n in args.n:
    print("n=%d begin" % n)
    # subset data. some sites might be non-segregating in the subsample.
    seg = [i for i, pos in enumerate(data[1]) if 
            any(h[i] != 0 for h in data[2][:n])
            and not all(h[i] == 1 for h in data[2][:n])]
    segdata = np.array([[ba[i] for i in seg] for ba in data[2][:n]])
    dsub = (data[0], data[1][seg], segdata)
    obs = smcpp.util.hmm_data_format(dsub, n, (0, 1)).astype('int32')
    # oo[n] = np.array([c1[1:] for c1 in obs for _ in range(c1[0])])
    # obs[obs[:, 1] != -1, 2:] = 0
    oo[n] = [smcpp.estimation_tools.thin_dataset([obs], [1])]
    oo[n], _ = smcpp.estimation_tools.break_long_spans(oo[n][0], 0)
    im = smcpp._smcpp.PyOnePopInferenceManager(n - 2, oo[n], hidden_states)
    im.theta = 2.0 * args.N0 * args.theta
    im.rho = 2.0 * args.estimated_rho * args.N0
    im.save_gamma = True
    im.model = model
    print("E step")
    im.E_step()
    print("E step done")
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
    full_gammas[n] = gamma
    print(gamma.mean(axis=1))
    gm[n] = scipy.ndimage.zoom(gamma, (1.0, 1. * width / args.L))
    maps[n] = np.argmax(gamma, axis=0)
    print("n=%d done" % n)

# np.savetxt(os.path.join(args.outdir, "hidden_states.txt"), hidden_states)
# np.savez_compressed(os.path.join(args.outdir, "posteriors.npz"), **gammadict)
# np.savez_compressed(os.path.join(args.outdir, "data.npz"), **datadict)

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
plt.set_cmap("magma")

hsmid = (hidden_states[1:] + hidden_states[:-1]) / 2.
hsmid[-1] = 2 * hsmid[-2]

tct_hs_out = (tct[None, :] - hsmid[:, None])**2

err = {n: (full_gammas[n] * tct_hs_out).sum() for n in full_gammas}
err_map = {n: tct_hs_out[np.argmax(full_gammas[n], axis=0), np.arange(args.L)].sum() for n in full_gammas}
with open(os.path.join(args.outdir, "mse.txt"), "wt") as f:
    for n in sorted(err):
        print("%d,%f,%f,%f,%f" % 
                (n, err[n], err[n] / err[2], 
                 err_map[n], err_map[n] / err_map[2]), file=f)
if not args.plot:
    sys.exit(0)

ai = 0
label_text   = [r"%i kb" % int(args.L / 40. * 100. * loc/width) for loc in plt.xticks()[0]]
mx = max([np.max(gm[g]) for g in gm])
sqer = {}
mapstep = {}
for n in sorted(gm):
    ax = axes[ai]
    ai += 1
    # im = ax.imshow(gm[n][::-1], extent=[0, width, -0.5, args.M - 0.5],aspect='auto', vmin=0.0, vmax=mx)
    img = NonUniformImage(ax, interpolation="bilinear")
    x = np.concatenate([[1.], np.cumsum(oo[n][0][:, 0])]) - 1
    xmax = x[-1]
    y = hidden_states
    y = (y[1:] + y[:-1]) / 2
    y[-1] = ymax = 2 * hidden_states[-2] # last val will be inf
    g = gammadict[str(n)][::-1]
    img.set_data(np.arange(0, xmax, xmax // width), y, gm[n])
    ax.images.append(img)
    # map_pos = scipy.ndimage.zoom(maps[n], 1. * width / args.L)
    am = np.argmax(g[::-1], axis=0)
    am2 = np.concatenate([[-1], am, [-1]])
    w = np.where((np.diff(am2) != 0))[0][1:] - 1
    map_pos = y[am[w]]
    y = np.concatenate([[0.], map_pos, map_pos[-1:]])
    x = np.concatenate([[0.], x[w], [xmax]])
    mapstep[n] = (x, y)
    ax.step(x, y, color="red", where="pre")
    s1 = StepFunction(x, y[:-1])
    ctx, cty = zip(*ct)
    ctx = np.concatenate([[0.], np.cumsum(ctx)])
    cty = np.concatenate([cty, cty[-1:]])
    ax.step(ctx, cty, where="pre", color=(0., 1., 0.))

    # add mutations as ticks
    a = oo[n][0]
    muts = np.where((a[:, 1] > 0) | (a[:, 2] > 0))
    mpos = np.cumsum(a[:, 0])[muts]
    for mp in mpos:
        ax.axvline(x=mp, ymin=.95, ymax=1, color="cyan")

    s2 = StepFunction(ctx, cty[:-1])
    sqer[n] = (s1 - s2)**2
    # ax.step(sqer[n]._x[:-1], sqer[n]._y, where="post", color="orange")
    # escore = (diff * gm[n]).sum() / diff.shape[1]
    ax.set_ylabel("n=%d" % n)
    ax.set_ylim([0, ymax])
    # ax.set_xticklabels(label_text)
    # corr = np.corrcoef(coal_times, maps[n])[0, 1]
    ax.text(xmax * 1.02, .8, "%.4f" % (err[n] / err[2]), rotation=-90, va='bottom', ha='right')
    # ax.text(xmax * 1.05, 1, "%.4f" % (err_map[n] / err_map[2]), rotation=-90, va='bottom', ha='right')

vmin = min(g.min() for g in gm.values())
vmax = max(g.max() for g in gm.values())

for img in ax.images:
    img.set_clim(vmin, vmax)

fig.colorbar(img, ax=axes.ravel().tolist(), shrink=0.5)

if args.plot:
    smcpp.plotting.save_pdf(fig, os.path.join(args.outdir, "plot.pdf"))
    plt.close(fig)
