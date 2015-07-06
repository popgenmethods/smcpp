# Complete example showing how to use the package for inference
from __future__ import division
import numpy as np
import scipy.optimize
import pprint
import multiprocessing
import sys
import itertools
from collections import Counter
import sys

import matplotlib
matplotlib.use('Agg')
import psmcpp.scrm, psmcpp.inference, psmcpp.bfgs, psmcpp._pypsmcpp, psmcpp.util, psmcpp.plotting

num_threads = 2
block_size = 10
num_samples = 1
np.set_printoptions(linewidth=120, precision=6, suppress=True)
N0 = 10000
rho = 1e-8
theta = 2e-8
L = int(1e7)
a = np.array([10, 2, 5])
b = np.array([1, 2, 4])
s = np.array([5000.0, 20000.0, 70000.]) / 25.0 / 2 * N0
true_parameters = (a, b, s)

n = 5
demography = psmcpp.scrm.demography_from_params((a, b, s / 2.0))
data = psmcpp.scrm.simulate(n, N0, theta, rho, L, demography, include_coalescence_times=True)

pd = {}
gm = {}
for nn in [2, 5]:
    # subset dat
    dsub = (data[0], data[1], data[2][:nn], data[3])
    obs = psmcpp.scrm.hmm_data_format(dsub, (0, 1))
    hidden_states = np.array([0., np.inf])
    im = psmcpp._pypsmcpp.PyInferenceManager(n - 2, [obs], hidden_states,
            2.0 * N0 * theta, 2.0 * N0 * rho * block_size,
            block_size, num_threads, num_samples)
    hidden_states = im.balance_hidden_states((a, b, s), 10)
    print("balanced hidden states", hidden_states)
    im = psmcpp._pypsmcpp.PyInferenceManager(n - 2, [obs], hidden_states,
            2.0 * N0 * theta, 2.0 * N0 * rho * block_size,
            block_size, num_threads, num_samples)
    im.setParams((a, b, s), False)
    im.Estep()
    gamma = im.gammas()[0]
    gm[nn] = gamma
    pd[nn] = psmcpp.inference.posterior_decode_score(0, 1, block_size, hidden_states, gamma, data[3])
    # Only care about first since they are all the same

def moving_average(a, n=10000) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=3, sharex=True, sharey=True, figsize=(20,6))
for ax, nn in zip(axes, sorted(gm)):
    ax.imshow(gm[nn], extent=[0,100,0,1], aspect='auto')
psmcpp.plotting.save_pdf(fig, "posterior_decoding_heatmap.pdf")

aoeu

fig, ax = psmcpp.plotting.pretty_plot()
for nn, color in zip(pd, psmcpp.plotting.palette[1:]):
    print(n, color)
    pd = np.mean([moving_average(test_posterior(n)) for _ in range(5)], axis=0)
    ax.plot(pd[::100], color=color, label=n)
psmcpp.plotting.save_pdf(fig, "posterior_decoding.pdf")

