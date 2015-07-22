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

import psmcpp.scrm, psmcpp.bfgs, psmcpp._pypsmcpp, psmcpp.util

num_threads = 10
block_size = 100
num_samples = 10
np.set_printoptions(linewidth=120, precision=6, suppress=True)

# 1. Generate some data. 
n = int(sys.argv[1])
N0 = 10000
rho = 1e-9
theta = 2.5e-8
L = int(2e7)

a = np.array([10., 1.])
b = np.array([10.01, 1.])
s = np.array([50000.0, 50000.0]) / 25.0 / (2 * N0)
true_parameters = (a, b, s)
print("true parameters")
print(np.array(true_parameters))
demography = psmcpp.scrm.demography_from_params((a * 2.0, b * 2.0, s))
data = psmcpp.scrm.simulate(n, N0, theta, rho, L, demography, False)
obs_pairs = [(2 * k, 2 * k + 1) for k in range(int(sys.argv[2]))]
obs_list = [psmcpp.scrm.hmm_data_format(data, cols) for cols in obs_pairs]
for ol0 in obs_list:
    obsfs = np.zeros([3, n - 1])
    for r, c1, c2 in ol0[ol0[:, 1:].sum(axis=1) > 0]:
        obsfs[c1, c2] += r
    obsfs /= L
    obsfs[0, 0] = 1. - obsfs.sum()
    print(" - Observed sfs:")
    print(obsfs)

# 4. Optimize this function
hidden_states = np.array([0., np.inf]) / 25.0 / (2 * N0)
im = psmcpp._pypsmcpp.PyInferenceManager(n - 2, obs_list, hidden_states,
        2.0 * N0 * theta, 2.0 * N0 * rho * block_size,
        block_size, num_threads, num_samples, 10)
hs1 = im.balance_hidden_states((a, b, s), 10)
em = np.arange(3 *  (n - 1), dtype=int).reshape([3, n - 1])
em[0] = em[2] = 0
em[1] = 1
im = psmcpp._pypsmcpp.PyInferenceManager(n - 2, obs_list, hs1,
        4.0 * N0 * theta, 4.0 * N0 * rho * block_size,
        block_size, num_threads, num_samples, 5, em)
# im.setDebug(True)
im.setParams((a, b, s), False)
im.Estep()
lam = 0.0
print("log likelikhood at true parameters:", im.loglik(lam))
im.setParams((a, b, s), ((0,0),))

s = np.array([10000.] * 10) / 25.0 / N0
K = len(s)
x0 = np.random.normal(5.0, 1.0, [2, K])
a = x0[0]
b = x0[1]

i = 0
im.seed = np.random.randint(0, sys.maxint)
llold = -np.inf
while i < 100:
    # Coordinate-y ascent
    c = i % K
    print("Optimizing coordinate %d..." % c)
    coords = [(0, c), (1, c)]
    if c == 0:
        bounds = ((0.1, 100), (0.1 * a[c + 1], 10.0 * a[c + 1]))
    elif c == K - 1:
        bounds = ((0.1 * b[c - 1], 10.0 * b[c - 1]), (0.1, 100.0))
    else:
        bounds = ((0.1 * b[c - 1], 10.0 * b[c - 1]), (0.1 * a[c + 1], 10.0 * a[c + 1]))
    print("Bounds are: %s" % str(bounds))
    def f(x):
        print("f", x)
        global a, b, s
        for coord, xx in zip(coords, x):
            x0[coord] = xx
        im.setParams((a, b, s), False)
        ret = [u for u, _ in im.Q(lam)]
        return -np.mean(ret)
    def fprime(x):
        print("fprime", x)
        global a, b, s
        for coord, xx in zip(coords, x):
            x0[coord] = xx
        im.setParams((a, b, s), coords)
        res = im.Q(lam)
        lls = np.array([ll for ll, jac in res])
        jacs = np.array([jac for ll, jac in res])
        #ret = -np.mean(jacs, axis=0)
        return (-np.mean(lls, axis=0), -np.mean(jacs, axis=0))
    res = scipy.optimize.fmin_l_bfgs_b(fprime, [x0[c] for c in coords], None, bounds=bounds, factr=1e12, disp=False)
    # res = scipy.optimize.fmin_tnc(f, x0, fprime, bounds=bounds)
    # res = scipy.optimize.fmin_bfgs(f, x0, fprime, disp=True)
    for coord, xx in zip(coords, res[0]):
        x0[coord] = xx
    print("************** ITERATION %d ***************" % i)
    print(x0.T)
    im.setParams((a, b, s), False)
    im.Estep()
    if i % 10 == 0:
        im.seed = np.random.randint(0, sys.maxint)
    ll = np.mean(im.loglik(lam))
    print(" - New loglik:" + str(ll))
    print(" - Old loglik:" + str(llold))
    llold = ll
    print(" - Estimated sfs:")
    print(im.sfs((a, b, s), 0, np.inf))
    print(" - Observed sfs:")
    print(obsfs)
    i += 1
