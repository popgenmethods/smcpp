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

import psmcpp.scrm, psmcpp.inference, psmcpp.bfgs, psmcpp._pypsmcpp, psmcpp.util

num_threads = 16
block_size = 20
num_samples = 20
np.set_printoptions(linewidth=120, precision=6, suppress=True)

# 1. Generate some data. 
n = int(sys.argv[1])
N0 = 10000
rho = 1e-8
theta = 1e-8
L = 10000000
a = np.log([10, 2, 5])
b = np.log([1, 2, 4])
s = np.array([5000.0, 20000.0, 70000.]) / 25.0 / N0
true_parameters = (a, b, s)
print("true parameters")
print(np.array(true_parameters))
demography = psmcpp.scrm.demography_from_params(true_parameters)
# Generate 3 datasets from this code by distinguishing different columns
data = psmcpp.scrm.simulate(n, N0, theta, rho, L, demography, include_coalescence_times=True)
obs_pairs = ((0, 1),(2, 3))[:int(sys.argv[2])]
obs_list = [psmcpp.scrm.hmm_data_format(data, cols) for cols in obs_pairs]

# 4. Optimize this function
hidden_states = np.array([0., 500., 1000., 5000., 10000., 50000., 100000., np.inf]) / 25.0 / N0
im = psmcpp._pypsmcpp.PyInferenceManager(n - 2, obs_list, hidden_states,
        4.0 * N0 * theta / 2.0, 4.0 * N0 * rho * block_size,
        block_size, num_threads, num_samples)
hs1 = im.balance_hidden_states((a, b, s), 10)
im = psmcpp._pypsmcpp.PyInferenceManager(n - 2, obs_list, hs1,
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

K = 3
# s = np.array([10000] * K) / 25.0 / N0
x0 = np.random.normal(3.0, 0.8, 2 * K)
a, b = x0.reshape((2, K))
print(x0)
bounds = ((np.log(0.10001), np.log(10000.0001)),) * K + ((np.log(0.1), np.log(10000)),) * K 
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
while i < 200:
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
for (l1, l2), gamma in zip(obs_pairs, im.gammas()):
    pd = psmcpp.inference.posterior_decode_score(l1, l2, block_size, hidden_states, gamma, data[3])
    print(pd)
