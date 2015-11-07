# Complete example showing how to use the package for inference
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

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 32})
import psmcpp.scrm, psmcpp.bfgs, psmcpp._pypsmcpp, psmcpp.util, psmcpp.plotting, psmcpp._newick

def norm_counter(c, nn): 
    s = sum(c.values()); 
    d = {k:np.array([1.*v/s]) for k,v in c.items()}
    ret = np.zeros([3, nn - 1])
    for k in d:
        ret[k] = d[k]
    return ret

def p_emit(obs, sfs):
    return (obs[None, ...] * np.log(sfs)).sum(axis=(1, 2))

def unpack(iterable):
    for span, x in iterable:
        for i in range(span):
            yield x

def pack(seq):
    iterable = iter(seq)
    x = next(iterable)
    i = 1
    for xp in iterable:
        if xp == x:
            i += 1
        else:
            yield (i, x)
            x = xp
            i = 1
    yield (i, x)

def subset_data(d, start, end):
    pnew = np.logical_and(start <= d[1], d[1] < end)
    inew = np.where(pnew)[0]
    return (int(end - start), d[1][pnew] - start,
            [np.array([h[i] for i in inew]) for h in d[2]],
            list(pack(itertools.islice(unpack(d[3]), start, end))))

block_size = 25
np.set_printoptions(linewidth=120, precision=6, suppress=True)
N0 = 10000
theta = 1.25e-8
rho = theta / 4.0
L = int(float(sys.argv[1]))
a = np.array([ 7.1,  7.1,  0.9,  7.1,  0.9,  7.1,  0.9])
b = np.array([ 7.1,  0.9,  7.1,  0.9,  7.1,  0.9,  0.9])
s = np.array([ 0.002   ,  0.006   ,  0.013   ,  0.109   ,  0.1     ,  1.77    ,  0.000002])
true_parameters = (a, b, s)
width = 2000
M = 32
G = np.zeros([M, L])

nns = [2, 5, 10, 25, 50, 100, 200]
n = max(nns)
demography = psmcpp.scrm.demography_from_params((a * 2.0, b * 2.0, s))
print(" ".join(map(str, demography)))
data = psmcpp.scrm.simulate(n, N0, theta, rho, L, demography, include_trees=True, seed=int(sys.argv[2]))

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

ct = [(c1, psmcpp._newick.tmrca(c2, "1", "2")) for c1, c2 in data[3]]

# Add noise
#ld = list(data)
#ld[2] = np.mod(ld[2] + (np.random.rand(*(data[2].shape)) < .01), 2)
#data = tuple(ld)

gm = {}
ims = {}
oo = {}
bks = {}
for nn in nns:
    # subset data. some sites might be non-segregating in the subsample.
    seg = [i for i, pos in enumerate(data[1]) if any(not h[i] for h in data[2][:nn])]
    segdata = np.array([[ba[i] for i in seg] for ba in data[2][:nn]])
    dsub = (data[0], data[1][seg], segdata, data[3])
    obs = psmcpp.scrm.hmm_data_format(dsub, (0, 1))
    oo[nn] = np.array([c1[1:] for c1 in obs for _ in range(c1[0])])
    hidden_states = np.array([0., 14.0])
    im = psmcpp._pypsmcpp.PyInferenceManager(nn - 2, [obs[:10]], hidden_states,
            4.0 * N0 * theta, 4.0 * N0 * rho, block_size, 10, [0])
    hidden_states = im.balance_hidden_states((a, b, s), M)
    hidden_states[-1] = 14.9
    em = np.arange(3 *  (nn - 1), dtype=int).reshape([3, nn - 1])
    em[0] = em[2] = 0
    em[1] = 1
    im = psmcpp._pypsmcpp.PyInferenceManager(nn - 2, [obs], hidden_states,
            4.0 * N0 * theta, 4.0 * N0 * rho, block_size, 1, [0], em)
    im.setParams((a, b, s), False)
    im.Estep()
    ims[nn] = im
    gamma = im.gammas()[0]
    bks[nn] = im.block_keys()[0]
    bb = 0
    for i, (_, d) in enumerate(bks[nn]):
        w = sum(d.values())
        G[:,bb:(bb+w)] = gamma[:,i:(i+1)]
        bb += w
    gm[nn] = scipy.ndimage.zoom(G, (1.0, 1. * width / L))

import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=len(nns), sharex=True, sharey=True, figsize=(25, 15))
# for i, ll in enumerate(((0, 1), (1, 2))[:2]):
coal_times = np.searchsorted(hidden_states, list(unpack(ct))) - 1
true_pos = scipy.ndimage.zoom(coal_times, 1. * width / L)
# axes[-1].step(range(width), true_pos)
#end i loop
#plt.set_cmap("cubehelix")
ai = 0
label_text   = [r"%i kb" % int(L / 40. * 100. * loc/width) for loc in plt.xticks()[0]]
mx = max([np.max(gm[g]) for g in gm])
for nn in sorted(gm):
    ax = axes[ai]
    ai += 1
    im = ax.imshow(gm[nn][::-1], extent=[0,width,-0.5,M-0.5],aspect='auto', vmin=0.0, vmax=mx)
    ax.step(range(width), true_pos, color=(0, 1., 1.))
    ax.set_ylabel("n=%d" % nn)
    ax.set_ylim([-0.5, M - 0.5])
    ax.set_xticklabels(label_text)
# axes[-1].set_ylabel("True hid. st.")
# axes[-1].set_ylim([-0.5, M - 0.5])
# axes[-1].xaxis.set_ticks(np.arange(0, width + 1, 100))
# fig.subplots_adjust(right=0.9)
# cbar_ax = fig.add_axes([0.92, 0.29, 0.02, 0.55])
# fig.colorbar(im, cax=cbar_ax)
psmcpp.plotting.save_pdf(fig, "posterior_decoding_heatmap.pdf")
plt.close(fig)

aoeu

fig, ax = psmcpp.plotting.pretty_plot()
for nn, color in zip(pd, psmcpp.plotting.palette[1:]):
    print(n, color)
    pd = np.mean([moving_average(test_posterior(n)) for _ in range(5)], axis=0)
    ax.plot(pd[::100], color=color, label=n)
psmcpp.plotting.save_pdf(fig, "posterior_decoding.pdf")
