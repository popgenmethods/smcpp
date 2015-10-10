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

np.set_printoptions(linewidth=120, suppress=True)

# 1. Generate some data. 
n = int(sys.argv[1])
L = int(float(sys.argv[2]))
N0 = 10000
theta = 2.5e-8
a = np.array([4., 3., 2., 1.])
b = a
s = np.array([20000.] * 4) / 25.0 / (2 * N0)
K = len(a)
true_parameters = (a, b, s)
print("true parameters")
print(np.array(true_parameters))
demography = psmcpp.scrm.demography_from_params((a * 2.0, b * 2.0, s))
observed_sfs = psmcpp.scrm.sfs(n, L, N0, theta, demography)
nz = observed_sfs != 0.0
print(observed_sfs)
onz = observed_sfs[nz]


def fprime(x):
    print(x)
    sfs, jac = psmcpp._pypsmcpp.sfs(n, (x, x, s), 0., np.inf, 4 * N0 * theta, jacobian=True)
    rsfs, rjac = [psmcpp._pypsmcpp.reduced_sfs(m) for m in sfs, jac]
    rsfs = rsfs[nz]
    rjac = rjac[nz]
    return (onz * np.log(onz / rsfs)).sum(axis=0), -(rjac * (onz / rsfs)[:, None]).sum(axis=0)[:K]

print("fprime at x0", fprime(a)[0])
print("sfs at x0")
sfs0 = psmcpp._pypsmcpp.sfs(n, (a, a, s), 0., np.inf, 4 * N0 * theta, jacobian=False)
print(psmcpp._pypsmcpp.reduced_sfs(sfs0))
print(sfs0)


bounds = ((0.10001, 100.0001),) * K

class MyBounds(object):
    def __init__(self, bounds):
        self.bounds = bounds
    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmin = bool(np.all(x >= self.bounds[:, 0]))
        tmax = bool(np.all(x <= self.bounds[:, 1]))
        return tmax and tmin

K = len(s)
x0 = np.random.normal(5.0, 1.0, K)

mybounds = MyBounds(np.array(bounds))
ret = scipy.optimize.basinhopping(fprime, x0, minimizer_kwargs={'method': 'L-BFGS-B', 'jac': True, 'bounds': bounds}, disp=True, stepsize=5.0)
print(ret)
aoeu

bounds = ((0.10001, 100.0001),) * K
res = scipy.optimize.fmin_l_bfgs_b(fprime, x0, None, bounds=bounds, disp=True, factr=1e2)
print("Observed sfs: ")
print(observed_sfs)
x = res[0]
print("Estimated sfs: ")
print(psmcpp._pypsmcpp.sfs(n, (x, x, s), 0., np.inf, 4 * N0 * theta, jacobian=False))
print("x*", x)
