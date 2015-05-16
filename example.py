# Complete example showing how to use the package for inference
import numpy as np
import psmcpp.scrm, psmcpp.inference
import pprint

NTHREADS=2

# 1. Generate some data. 
# We'll focus on the **simplest** # case of inferring a 
# **constant** demography to start.
n = 10
N0 = 10000
theta = rho = 1e-8
L = 1000000
data = psmcpp.scrm.simulate(n, N0, theta, rho, L) # no demography
# Generate 3 datasets from this code by distinguishing different 
# columns
obs_list = [psmcpp.scrm.hmm_data_format(data, cols) for cols in ((0, 1), (2, 3), (4, 5))]

# 2. Set up some model parameters
# We'll use a 3-period model just to see how things work.
# Hopefully the inference code will make the periods 
# look pretty equal to one another.
sqrt_a = [1.0, 1.0, 1.0]
a_scale = 1.
b = [0.0001, 0.0001, 0.0000001]
b_scale = 1. / .0001
sqrt_s = [0.0, .5, 1.0]
s_scale = 1.
# We'll use 10 hidden states
hidden_states = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, np.inf])

# 3. Compute the log-likelihood and derivatives with respect
# to the data.
S = 1000
M = 20
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
K = len(sqrt_a)
def logp(x):
    x = tuple(x)
    ax = np.array(x).reshape((3, K))
    if x not in logp._memo:
        sqrt_a, b, sqrt_s = ax
        ret = psmcpp.inference.logp(
                sqrt_a, b, sqrt_s, # Model parameters
                S, M, # Parameters which govern accuracy with which
                # the SFS is numerically computed.
                n,
                obs_list, # List of the observations datasets we prepared above
                N0 * theta, N0 * rho, # Same parameters as above
                hidden_states,
                numthreads=NTHREADS, # Using multiple threads speeds everything up.
                viterbi=True,
                reg_a=10, reg_b=10, reg_s=10
                )
        logp._memo[x] = ret
    return logp._memo[x]
logp._memo = {}

scales = np.array([a_scale, b_scale, s_scale])

def rescale(x):
    ax = np.array(x).reshape((3, K))
    ax = ax / scales[:, None]
    return ax.reshape(3 * K)

def f(x):
    print("x:")
    print(np.array(x).reshape(3, K))
    print("x unscaled:")
    ax = np.array(rescale(x)).reshape(3, K)
    print(ax)
    lp = logp(ax.reshape(3 * K))
    print("negll", -lp[0])
    print("df/dx")
    print(lp[1])
    print("df/dx unscaled")
    dfdx = rescale(lp[1])
    print(dfdx)
    print("viterbi")
    pprint.pprint(dict(lp[2]))
    return (-lp[0], -dfdx)

def fprime(x):
    dx = logp(rescale(x))[1]
    dxv = dx.view()
    dxv.shape = (3, K)
    # Fix last b to be (almost) piecewise constant
    dxv[1, -1] = 0.0
    return -np.array(rescale(dx))

np.set_printoptions(suppress=True)
x = (np.array([sqrt_a, b, sqrt_s]) * scales[:, None]).reshape(3 * K)
alpha = 0.01
while True:
    print(x)
    print(f(x))
    x += fprime(x) / np.linalg.norm(x) * alpha
