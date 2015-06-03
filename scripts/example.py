# Complete example showing how to use the package for inference
from __future__ import division
import numpy as np
import scipy.optimize
import pprint

import psmcpp.scrm, psmcpp.inference, psmcpp.bfgs, psmcpp._pypsmcpp

NTHREADS=1
np.random.seed(1000)

# 1. Generate some data. 
# We'll focus on the **simplest** # case of inferring a 
# **constant** demography to start.
n = 15;
N0 = 10000
theta = rho = 1e-8
L = 1000000
demography = ['-eN', 0.0, 4.0, '-eN', 0.5, 0.5]
# sfs = psmcpp.scrm.sfs(n, 1000000, N0, theta, demography) # no demography
# Generate 3 datasets from this code by distinguishing different 
# columns
data = psmcpp.scrm.simulate(n, N0, theta, rho, L, demography) # no demography
obs_list = [psmcpp.scrm.hmm_data_format(data, cols) for cols in ((0, 1), (2, 3), (4, 5), (6, 7))]
# obs_list = [psmcpp.scrm.hmm_data_format(data, cols) for cols in ((0,1),(2,3))]
# obs_list = [psmcpp.scrm.hmm_data_format(data, cols) for cols in ((0,1),(2,3))]

# 2. Set up some model parameters
# We'll use a 3-period model just to see how things work.
# Hopefully the inference code will make the periods 
# look pretty equal to one another.
a = np.log([1.0] * 10)
b = np.log([1.5] * 10)
s = [0.1] * 10
# We'll use 10 hidden states
hidden_states = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, np.inf])

# 3. Compute the log-likelihood and derivatives with respect
# to the data.
S = 100000
M = 10
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
    ax = np.array(x).reshape((2, K))
    if key not in logp._memo:
        a, b = ax
        print("calling logp with", a, b, jacobian)
        ret = psmcpp.inference.loglik(
                (a, b, s),
                n - 2,
                S, M, # Parameters which govern accuracy with which
                # the SFS is numerically computed.
                obs_list, # List of the observations datasets we prepared above
                hidden_states,
                N0 * rho, N0 * theta, # Same parameters as above
                reg=0.0,
                numthreads=NTHREADS, # Using multiple threads speeds everything up.
                jacobian=jacobian
                ) 
        logp._memo[key] = ret
    return logp._memo[key]
logp._memo = {}

def sfs_l2(x, jacobian):
    key = tuple((tuple(x), jacobian))
    ax = np.array(x).reshape((2, K))
    if key not in logp._memo:
        a, b = ax
        ret = psmcpp._pypsmcpp.pretrain(
                (a, b, s),
                n - 2,
                S, M, # Parameters which govern accuracy with which
                # the SFS is numerically computed.
                sfs, # List of the observations datasets we prepared above
                0.0, 
                NTHREADS, # Using multiple threads speeds everything up.
                N0 * theta,
                jacobian=jacobian
                )
        logp._memo[key] = ret
    return logp._memo[key]
logp._memo = {}

def f(x):
    ret = sfs_l2(x, False)
    print("f(%s) = %f" % (str(x), ret))
    return ret
def fprime(x):
    ret = sfs_l2(x, True)[1][:2]
    print("f'(%s) = %s" % (str(x), str(ret)))
    return ret.reshape((2 * K))

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

x0 = np.array([a, b]).flatten()
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
f(x0)
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
