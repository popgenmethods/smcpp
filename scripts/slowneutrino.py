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

import psmcpp.scrm, psmcpp._pypsmcpp

np.set_printoptions(linewidth=120)

# 1. Generate some data. 
n = int(sys.argv[1])
seed = int(sys.argv[2])
N0 = 10000
theta = 2.5e-8
L = int(1e5)
a = np.array([10., 1., .1, 2.])
b = np.array([1., 1., .1, 2.])
s = np.array([5000.0, 50000.0, 10000., 10000.]) / 25.0 / (2 * N0)
true_parameters = (a, b, s)
print("true parameters")
print(np.array(true_parameters))
demography = psmcpp.scrm.demography_from_params((a * 2.0, b * 2.0, s))
observed_sfs = psmcpp.scrm.sfs(n, L, N0, theta, demography, seed=seed) / L
eps = 1e-2
demography2 = psmcpp.scrm.demography_from_params(((a + [eps, 0, 0, 0]) * 2.0, b * 2.0, s))
observed_sfs2 = psmcpp.scrm.sfs(n, L, N0, theta, demography2, seed=seed) / L
# sfs, rsfs, jac, rjac = psmcpp._pypsmcpp.sfs((a, b, s), n - 2, 200, 0., np.inf, 2, 4 * N0 * theta, jacobian=True, seed=seed)
d = (observed_sfs2 - observed_sfs) / eps
print(np.array([d, observed_sfs]).T)
aoeu

K = a.shape[0]

def f(x):
    # print("f", x, recompute)
    a, b = x.reshape((2, K))
    _, rsfs = psmcpp._pypsmcpp.sfs((a, b, s), n - 2, 200, 0., np.inf, 2, 4 * N0 * theta, jacobian=False, seed=1) 
    kl = (observed_sfs * np.log(observed_sfs / rsfs)).sum()
    return kl

def fprime(x):
    a, b = x.reshape((2, K))
    _, rsfs, jac = psmcpp._pypsmcpp.sfs((a, b, s), n - 2, 200, 0., np.inf, 2, 4 * N0 * theta, jacobian=True, seed=1) 
    return -((observed_sfs / rsfs)[:, :, None] * jac).sum(axis=(0, 1))

aoeu

def loglik(x):
    a, b = x.reshape((2, K))
    im.setParams((a, b, s), False)
    ll = im.loglik(lam)
    # print('ll', ll)
    return -np.mean(ll)

K = len(s)
# s = np.array([2000, 2000, 3000, 5000, 20000, 50000, 50000]) / 25.0 / N0
x0 = np.random.normal(5.0, 1.0, 2 * K)

bounds = ((0.10001, 100.0001),) * K + ((0.1, 100),) * K 

# # Pretrain
# def pretrain(x):
#     a, b = x.reshape((2, K))
#     # sfs, _= psmcpp._pypsmcpp.sfs((a, b, s), n - 2, num_samples, 0., np.inf, num_threads, 2 * N0 * theta, jacobian=True, seed=1)
#     im.seed = 1
#     sfs = im.sfs((a, b, s), 0, np.inf)
#     print(x.reshape((2, K)))
#     C = (obsfs * np.log(sfs))
#     # print(C)
#     return -C.sum()
# 
# def dpretrain(x):
#     a, b = x.reshape((2, K))
#     im.seed = 1
#     sfs, jac = im.sfs((a, b, s), 0, np.inf, True)
#     # sfs = im.sfs((a, b, s), 0, np.inf)
#     dj = (obsfs[:, :, None] * jac / sfs[:, :, None])
#     # print(C)
#     return -dj.sum(axis=(0, 1))[:(2 * K)]

# print scipy.optimize.check_grad(pretrain, dpretrain, x0)
# aoeu
# scipy.optimize.minimize(pretrain, x0, method="L-BFGS-B", bounds=bounds)
# print("best ll", pretrain(np.array([a, b]).flatten()))
# scipy.optimize.basinhopping(pretrain, x0, stepsize=1.0, 
#         minimizer_kwargs={"method": "L-BFGS-B", "bounds": bounds, "jac": dpretrain}, 
#         disp=True)

i = 0
im.seed = 1234
im.setParams((a, b, s), False)
im.Estep()
llold = loglik(x0)
while i < 100:
    res = scipy.optimize.fmin_l_bfgs_b(f, x0, fprime, bounds=bounds, factr=1e14, disp=False)
    # res = scipy.optimize.fmin_tnc(f, x0, fprime, bounds=bounds)
    # res = scipy.optimize.fmin_bfgs(f, x0, fprime, disp=True)
    xlast = x0.reshape((2, K))
    x0 = res[0].reshape((2, K))
    print("************** ITERATION %d ***************" % i)
    print(x0.T)
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
    print(im.sfs((a, b, s), 0, np.inf))
    print(" - Observed sfs:")
    print(obsfs)
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
