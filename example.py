# Complete example showing how to use the package for inference
import numpy as np
import psmcpp.scrm, psmcpp.inference

# 1. Generate some data. 
# We'll focus on the **simplest** # case of inferring a 
# **constant** demography to start.
n = 10
N0 = 10000
theta = rho = 1e-8
L = 100000
data = psmcpp.scrm.simulate(n, N0, theta, rho, L) # no demography
# Generate 3 datasets from this code by distinguishing different 
# columns
obs_list = [psmcpp.scrm.hmm_data_format(data, cols) for cols in ((0, 1), (2, 3), (4, 5))]

# 2. Set up some model parameters
# We'll use a 3-period model just to see how things work.
# Hopefully the inference code will make the periods 
# look pretty equal to one another.
sqrt_a = 1. / np.sqrt([1000.0, 1000.0, 1000.0])
a_scale = 1000. * np.sqrt(1000.)
b = np.array([0.0001, 0.0001, 0.0001]) 
b_scale = 1000. / .0001
sqrt_s = np.sqrt([0.0, .5, 1.0])
s_scale = 1000.
# We'll use 10 hidden states
hidden_states = np.append(np.linspace(0.0, 5.0, 5), np.inf)

# 3. Compute the log-likelihood and derivatives with respect
# to the data.
S = 2000
M = 25 
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
                theta, rho, # Same parameters as above
                hidden_states,
                numthreads=16 # Using multiple threads speeds everything up.
                )
        logp._memo[x] = ret
    print(np.array(x).reshape(3, K), logp._memo[x])
    return logp._memo[x]
logp._memo = {}

scales = np.array([a_scale, b_scale, s_scale])

def rescale(x):
    ax = np.array(x).reshape((3, K))
    ax /= scales
    return tuple(ax.reshape(3 * K))

def f(x):
    return logp(rescale(x))[0]

def fprime(x):
    dx = logp(rescale(x))[1]
    return rescale(dx)

x0 = (np.array([sqrt_a, b, sqrt_s]) * scales).reshape(3 * K)
ret = scipy.optimize.fmin_bfgs(f, x0, fprime, disp=True)
print(ret)

