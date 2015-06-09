# Complete example showing how to use the package for inference
from __future__ import division
import numpy as np
import scipy.optimize
import pprint
import multiprocessing

import psmcpp.scrm, psmcpp.inference, psmcpp.bfgs, psmcpp._pypsmcpp

NTHREADS=8
np.random.seed(1000)
np.set_printoptions(suppress=True)

# 1. Generate some data. 
# We'll focus on the **simplest** # case of inferring a 
# **constant** demography to start.
n = 30
N0 = 10000
rho = 1e-8
theta = 1e-8
L = 10000000
demography = ['-eN', 0.0, 8.0, '-eG', 0.0, -0.01, '-eN', 0.5, 0.5]

# Generate 3 datasets from this code by distinguishing different 
# columns
# data = psmcpp.scrm.simulate(n, N0, theta, rho, L, demography) # no demography
# obs_list = [psmcpp.scrm.hmm_data_format(data, cols) for cols in ((0, 1), (2, 3))]

Lsfs = 100000
p = multiprocessing.Pool(10)
tasks = [p.apply_async(psmcpp.scrm.sfs, (n, Lsfs, N0, theta, demography)) for _ in range(10)]
sfs = np.mean([t.get() for t in tasks], axis=0)
print(sfs)

# aoeu
# sfs = np.array([997463.75, 1230.625, 475., 251.125, 147., 93.375, 67., 
#     47.875, 36.125, 27., 26.625, 22.125, 18.75, 21.125, 14.875, 12.75, 
#     12.375, 10.75, 11.625, 10.125]) * 8
# obs_list = [psmcpp.scrm.hmm_data_format(data, cols) for cols in ((0,1),(2,3))]
# obs_list = [psmcpp.scrm.hmm_data_format(data, cols) for cols in ((0,1),(2,3))]

# 2. Set up some model parameters
# We'll use a 3-period model just to see how things work.
# Hopefully the inference code will make the periods 
# look pretty equal to one another.
K = 2
a = np.array([1.] * K)
b = np.array([1.1] * K)
s = np.array([1.0] * K) * 2
hidden_states = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, np.inf])
x0 = np.array([a, b, s]).flatten()
# x0 = 1. / (2. + np.random.random(8) * 1e-6)
# print(x0)

# 3. Compute the log-likelihood and derivatives with respect
# to the data.
S = 100000
M = 200
# logp, jacobian = psmcpp.inference.logp(
#         sqrt_a, b, sqrt_s, # Model parameters
#         S, M, # Parameters which govern accuracy with which
#               # the SFS is numerically computed.
#         n,
#         obs_list, # List of the observations datasets we prepared above
#         theta, rho, # Same parameters as above
#         hidden_states,
#         numthreads=8 # Using multiple threads speeds everything up.
#         )
# print(logp)
# print(jacobian)

# 4. Optimize this function
K = len(a)
def logp(x, jacobian):
    key = tuple((tuple(x), jacobian))
    ax = np.array(x).reshape((3, K))
    if key not in logp._memo:
        a, b, s = ax
        ret = psmcpp.inference.loglik(
                (a, b, s),
                n - 2,
                S, M, # Parameters which govern accuracy with which
                # the SFS is numerically computed.
                obs_list, # List of the observations datasets we prepared above
                hidden_states,
                4. * N0 * rho, 4. * N0 * theta / 2., # Same parameters as above
                reg=10.0,
                numthreads=NTHREADS, # Using multiple threads speeds everything up.
                jacobian=jacobian
                ) 
        logp._memo[key] = ret
    return logp._memo[key]
logp._memo = {}

def sfs_kl(x, jacobian):
    key = tuple((tuple(x), jacobian))
    ax = np.array(x).reshape((3, K))
    if key not in logp._memo:
        a, b, s = ax
        ret = psmcpp._pypsmcpp.pretrain(
                (a, b, s),
                n - 2,
                S, M, # Parameters which govern accuracy with which
                # the SFS is numerically computed.
                sfs, # List of the observations datasets we prepared above
                10.0,
                NTHREADS, # Using multiple threads speeds everything up.
                4 * N0 * theta / 2,
                jacobian=jacobian
                )
        logp._memo[key] = ret
    return logp._memo[key]
logp._memo = {}

def f(x):
    ret = -sfs_kl(x, False)
    print("f(%s) = %f" % (str(x.reshape((3, K))), ret))
    return ret
def fprime(x):
    ret = -sfs_kl(x, True)[1]
    print("f'(%s) = %s" % (str(x), str(ret)))
    return ret.reshape((3 * K))

bounds = ((0.01, 1000.),) * 2 * K + ((0.1, 1.0),) * K
res = scipy.optimize.minimize(f, x0, jac=fprime, bounds=bounds, method="L-BFGS-B")
print(res)
aeou


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
#    aoeu
#    print(ret)
#    xk, xlast = xk - ret.alpha0 * pk + 0.1 * (xk - xlast), xk

res = scipy.optimize.minimize(f, x0, jac=fprime, bounds=bounds, method="TNC")
print(res)
print(res.x.reshape((2, K)))
aoeu

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
