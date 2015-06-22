# Complete example showing how to use the package for inference
from __future__ import division
from random import randint
from time import sleep
import numpy as np
import scipy.optimize
import pprint
import multiprocessing
import sys
import itertools
from collections import Counter
import argparse
import sys

import psmcpp.scrm, psmcpp.inference, psmcpp.bfgs, psmcpp._pypsmcpp

num_threads = 8
block_size = 20
num_samples = 25
np.set_printoptions(linewidth=120, precision=6, suppress=True)

# 1. Generate some data. 
n = int(sys.argv[1])
em_iters = int(sys.argv[2])
N0 = 3500
rho = 1e-8
theta = 1e-8
L = 10000000
a = np.log([1.0, 10**-4, 0.1])
b = np.log([10**-2, 10**-4, 0.1])
s = np.array([1.0, 1.0, 1.0])
true_parameters = (a, b, s)
print("true parameters")
print(np.array(true_parameters))
demography = psmcpp.scrm.demography_from_params(true_parameters)
# Generate 3 datasets from this code by distinguishing different columns
data = psmcpp.scrm.simulate(n, N0, theta, rho, L, demography, include_trees=False)
obs_list = [psmcpp.scrm.hmm_data_format(data, cols) for cols in ((0,1),)]

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return itertools.izip_longest(fillvalue=fillvalue, *args)

# Format trees
def tree_obs_iter(trees):
    fs = frozenset([1, 2])
    for sec, d in trees:
        for i in range(sec):
            yield d[fs]

def trueB(trees, hs):
    tb = []
    M = len(hs) - 1
    for block in grouper(tree_obs_iter(data[3]), block_size):
        a = np.zeros([M, 1])
        c = Counter(block)
        s = sum(c.values())
        for k in c:
            ip = np.searchsorted(hs, k) - 1
            a[ip] = 1. * c[k] / s
        tb.append(a)
    return np.array(tb).T

# 4. Optimize this function
hidden_states = np.array([0., 1.0, 2.0, 3.0, 4.0, np.inf])
im = psmcpp._pypsmcpp.PyInferenceManager(n - 2, obs_list, hidden_states,
        4.0 * N0 * theta / 2.0, 4.0 * N0 * rho * block_size,
        block_size, num_threads, num_samples)
lam = 0.0

def f(x):
    # print("f", x, recompute)
    a, b = x.reshape((2, K))
    im.setParams((a, b, s), False)
    ret = [a for a, b in im.Q(lam)]
    # print('Q', ret)
    return -np.mean(ret)

def fprime(x, recompute=False):
    a, b = x.reshape((2, K))
    im.setParams((a, b, s), True)
    res = im.Q(lam)
    jacs = np.array([jac[:2] for ll, jac in res])
    ret = -np.mean(jacs, axis=0).reshape((2 * K))
    # print('fprime', ret)
    return ret
    # print("f'(%s) = %s" % (str(x.reshape((2, K))), str(ret)))

def loglik(x):
    a, b = x.reshape((2, K))
    im.setParams((a, b, s), False)
    ll = im.loglik(lam)
    # print('ll', ll)
    return -np.mean(ll)

K = 6
s = [0.5] * K
x0 = np.random.normal(3.0, 0.8, 2 * K)
a, b = x0.reshape((2, K))
print(x0)
bounds = ((np.log(0.000001), np.log(10.0001)),) * K + ((np.log(.000002), np.log(10.0002)),) * K 
i = 0
# hs1 = im.balance_hidden_states((a, b, s), 5)
# im = psmcpp._pypsmcpp.PyInferenceManager(n - 2, obs_list, hs1,
#         4.0 * N0 * theta / 2.0, 4.0 * N0 * rho * block_size,
#         block_size, num_threads, num_samples)
im.setParams((a, b, s), False)
im.Estep()
# print(im.gammas())
# tb = trueB(data[3], hs1)
# print(tb)
llold = loglik(x0)
while i < em_iters:
    res = scipy.optimize.fmin_l_bfgs_b(f, x0, fprime, bounds=bounds, factr=1e14, disp=False)
    # res = scipy.optimize.fmin_bfgs(f, x0, fprime, disp=True)
    xlast = x0.reshape((2, K))
    x0 = res[0].reshape((2, K))
    print("************** ITERATION %d ***************" % i)
    print(x0)
    print(np.linalg.norm(xlast - x0) / np.linalg.norm(x0))
    a, b = x0
    im.setParams((a, b, s), True)
    # im.setDebug(True)
    im.Estep()
    if i % 10 == 0:
        im.seed = np.random.randint(0, sys.maxint)
    # print(im.gammas())
    ll = loglik(x0)
    print(" - New loglik:" + str(ll))
    print(" - Old loglik:" + str(llold))
    print(" - Estimated sfs:")
    llold = ll
    #if i % 10 == 0:
    #    hs1 = im.balance_hidden_states((a, b, s), 5)
    #    im = psmcpp._pypsmcpp.PyInferenceManager(n - 2, obs_list, hs1,
    #            4.0 * N0 * theta / 2.0, 4.0 * N0 * rho * block_size,
    #            block_size, num_threads, num_samples)
    #    im.setParams((a, b, s), True)
    #    im.Estep()
    #    print("new hidden states", hs1)
    i += 1
sleep(randint(1,20))
open("output.%d.txt" % n, "a").write(str(x0) + "\n")
