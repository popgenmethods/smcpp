# Complete example showing how to use the package for inference
from __future__ import division
import numpy as np
import psmcpp.scrm, psmcpp.inference
import pprint

np.random.seed(1)

NTHREADS=2

# 1. Generate some data. 
# We'll focus on the **simplest** # case of inferring a 
# **constant** demography to start.
n = 6
N0 = 10000
theta = rho = 1e-8
L = 10000
demography = ['-eN', 0.5, 0.5]
data = psmcpp.scrm.simulate(n, N0, theta, rho, L, demography) # no demography
# Generate 3 datasets from this code by distinguishing different 
# columns
# obs_list = [psmcpp.scrm.hmm_data_format(data, cols) for cols in ((0, 1), (2, 3), (4, 5), (6, 7))]
obs_list = [psmcpp.scrm.hmm_data_format(data, cols) for cols in ((0,1),)]

# 2. Set up some model parameters
# We'll use a 3-period model just to see how things work.
# Hopefully the inference code will make the periods 
# look pretty equal to one another.
a = [1.0, 2.0]
b = [1.5, 2.5]
s = [0.2, 0.5]
# We'll use 10 hidden states
hidden_states = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, np.inf])

# 3. Compute the log-likelihood and derivatives with respect
# to the data.
S = 1000
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
import scipy.optimize
K = len(a)
def logp(x, jacobian):
    key = tuple((tuple(x), jacobian))
    ax = np.array(x).reshape((3, K))
    if key not in logp._memo:
        a, b, s = ax
        ret = psmcpp.inference.loglik(
                (a, b, s),
                n,
                S, M, # Parameters which govern accuracy with which
                # the SFS is numerically computed.
                obs_list, # List of the observations datasets we prepared above
                hidden_states,
                N0 * rho, N0 * theta, # Same parameters as above
                reg=1.0,
                numthreads=NTHREADS, # Using multiple threads speeds everything up.
                jacobian=jacobian
                )
        logp._memo[key] = ret
    return logp._memo[key]
logp._memo = {}

def f(x):
    ret = -logp(x, False)
    print("f(%s) = %f)" % (str(x), ret))
    return ret
def fprime(x):
    ret = -logp(x, True)[1]
    print("f'(%s) = %s)" % (str(x), str(ret)))
    return ret

x0 = np.array([a, b, s])
res = scipy.optimize.minimize(f, x0, jac=fprime)
print(res)

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
