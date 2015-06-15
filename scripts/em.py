# Complete example showing how to use the package for inference
from __future__ import division
import numpy as np
import scipy.optimize
import pprint
import multiprocessing
import sys

import psmcpp.scrm, psmcpp.inference, psmcpp.bfgs, psmcpp._pypsmcpp

num_threads = 1
np.set_printoptions(linewidth=120, precision=6, formatter={'float': lambda x: "%g" % x})

# 1. Generate some data. 
# We'll focus on the **simplest** # case of inferring a 
# **constant** demography to start.
n = 10
N0 = 10000
rho = 1e-8
theta = 1e-8
L = 1000000
demography = ['-eN', 0.0, 8.0, '-eG', 0.0, -0.5, '-eN', 3, 0.5]

# Generate 3 datasets from this code by distinguishing different 
# columns
data = psmcpp.scrm.simulate(n, N0, theta, rho, L, demography) # no demography
# obs_list = [psmcpp.scrm.hmm_data_format(data, cols) for cols in ((0, 1), (2, 3), (4, 5), (6, 7))]
obs_list = [psmcpp.scrm.hmm_data_format(data, cols) for cols in ((0,1),)]
obsfs = np.zeros([3, n + 1])
ol0 = obs_list[0]
for r, a, b in ol0[ol0[:, 1:].sum(axis=1) > 0]:
    obsfs[a, b] += r
obsfs /= L
obsfs[0, 0] = 1. - obsfs.sum()
print("observed sfs")
print(obsfs)


# 2. Set up some model parameters
# We'll use a 3-period model just to see how things work.
# Hopefully the inference code will make the periods 
# look pretty equal to one another.
K = 5
a = np.array([1.] * K)
b = np.array([1.1] * K)
s = np.array([1.0] * K) * 2
# hidden_states = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, np.inf])
hidden_states = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 2.0, np.inf])
x0 = np.array([a, b, s]).flatten()

# 3. Compute the log-likelihood and derivatives with respect
# to the data.
num_samples = 20
block_size = 20

# 4. Optimize this function
K = len(a)
im = psmcpp._pypsmcpp.PyInferenceManager(n - 2, obs_list, hidden_states, 
        4.0 * N0 * theta / 2.0, 4.0 * N0 * rho, block_size, num_threads, num_samples)

def f(x):
    # print("f", x, recompute)
    a, b = x.reshape((2, K))
    im.setParams((a, b, s), False)
    ret = [a for a, b in im.Q(0.0)]
    print('Q', ret)
    return -np.mean(ret)

def fprime(x, recompute=False):
    a, b = x.reshape((2, K))
    im.setParams((a, b, s), True)
    res = im.Q(0.0)
    jacs = np.array([jac[:2] for ll, jac in res])
    return -np.mean(jacs, axis=0).reshape((2 * K))
    # print("f'(%s) = %s" % (str(x.reshape((2, K))), str(ret)))

def loglik(x):
    a, b = x.reshape((2, K))
    im.setParams((a, b, s), False)
    ll = im.loglik(0.0, False)
    print('ll', ll)
    return -np.mean(ll)

#bounds = ((0.10001, 100.0001),) * K + ((0.1, 100),) * K + ((0.1, 0.5),) * K
x0 = np.random.normal(3.0, 0.8, 2 * K)
print(x0)
bounds = ((0.10001, 1000.0001),) * K + ((0.1, 1000),) * K 
i = 0
a, b = x0.reshape((2, K))
im.setParams((a, b, s), False)
im.Estep(False)
llold = loglik(x0)
im.setParams((a, b, s), True)
im.Estep(True)
while True:
    res = scipy.optimize.fmin_l_bfgs_b(f, x0, fprime, bounds=bounds, factr=1e14, disp=False)
    x0 = res[0].reshape((2, K))
    print("************** ITERATION %d ***************" % i)
    print(x0)
    a, b = x0
    for ad in (True, False):
        im.setParams((a, b, s), ad)
        im.Estep(ad)
    ll = loglik(x0)
    print(" - New loglik:" + str(ll))
    print(" - Old loglik:" + str(llold))
    print(" - Estimated sfs:")
    llold = ll
    i += 1
    # sfs = psmcpp._pypsmcpp.sfs(
    #         (x0.reshape((2, K))[0], x0.reshape((2, K))[1], s), 
    #         n - 2, 
    #         num_samples, 0, np.inf, NTHREADS,
    #         4 * N0 * theta / 2.)
    # print(sfs[0])
    # print(" - Observed sfs:")
    # print(obsfs)

def f(x):
    # ret = -sfs_kl(x, False)
    ret = -logp(x, False)
    print("f(%s) = %f" % (str(x.reshape((3, K))), ret))
    return ret
def fprime(x):
    # ret = -sfs_kl(x, True)[1]
    ret = -logp(x, True)[1]
    print("f'(%s) = %s" % (str(x), str(ret)))
    return ret.reshape((3 * K))
minimizer_kwargs = {"method": "L-BFGS-B", 'jac': fprime, 'bounds': bounds}
def accept_test(f_new, x_new, f_old, x_old):
    return np.all([x_new < 1000., x_new > .01])
res = scipy.optimize.basinhopping(
        f, x0, minimizer_kwargs=minimizer_kwargs, niter=100,
        accept_test=accept_test, disp=True)
print(res)

#xk = x0
#xlast = 0
#while True:
#    pk = -fprime(xk)
#    ret = scipy.optimize.line_search(f, fprime, xk, pk, amax=2.0)
#    print(ret)
#    xk, xlast = xk - ret.alpha0 * pk + 0.1 * (xk - xlast), xk

res = scipy.optimize.minimize(f, x0, jac=fprime, bounds=bounds, method="TNC")
print(res)
print(res.x.reshape((2, K)))

# x = x0
# xlast = 0
# I = np.eye(2 * K)
# while True:
#     y = f(x)
#     dy = fprime(x)
#     k = np.random.randint(2 * K)
#     alpha = 0.5
#     while f(x - alpha * dy[k] * I[k]) > 0.9 * y:
#         alpha *= 1.5
#     x, xlast = x - alpha * dy + 0.2 * (x - xlast), x

# print(scipy.optimize.check_grad(f, fprime, x0))

# res = scipy.optimize.minimize(f, x0, jac=fprime)
# print(res)

def f(x):
    ret = logp(x, False) / L * 1000.
    print("f(%s) = %f" % (str(x), ret))
    return ret
def fprime(x):
    ret = logp(x, True)[1][:2] / L * 1000.
    print("f'(%s) = %s" % (str(x), str(ret)))
    return ret.reshape((2 * K))
res = scipy.optimize.minimize(f, x0, jac=fprime)

#i = 20 
#while True:
#    alpha = 0.5 / np.sqrt(i)
#    i += 1
#    y, dy, vit = f(x)
#    print("objective: %f" % y)
#    print(dy.reshape((3, K)))
#    print(x.reshape((3, K)))
#    print("viterbi", dict(vit))
#    x += dy / np.linalg.norm(dy) * alpha
