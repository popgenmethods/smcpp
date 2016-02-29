import numpy as np
from multiprocessing import Pool
import pytest

# import warnings
# warnings.filterwarnings("error")
import numpy as np

from fixtures import *
from _pypsmcpp import Demography

ls = np.linspace(0.0, 10.0, 20)

def test_R_flat():
    u = np.log([1.0 - .01, 1.0 - .01])
    v = np.log([1.0 - .01, 1.0 - .01])
    s = np.log([0.0, .3])
    d = Demography(u, v, s)
    d.print_debug()
    for t in ls:
        assert abs(d.R(t) - t) < 1e-8
        assert abs(d.inverse_rate(t) - t) < 1e-8
 
def test_R_constant(constant_demo_1):
    for t in ls:
        assert(abs(constant_demo_1.R(t)) - t) < 1e-8

def test_R_constant_1000(constant_demo_1000):
    for t in ls:
        assert(abs(constant_demo_1000.R(t)) - 1000. * t) < 1e-8

def test_Rinv_constant(constant_demo_1, constant_demo_1000, demo):
    for d in constant_demo_1, constant_demo_1000, demo:
        for t in ls:
            assert abs(d.R(d.inverse_rate(t)) - t) < 1e-8
            assert abs(d.inverse_rate(d.R(t)) - t) < 1e-8

def test_R_numerical(constant_demo_1, constant_demo_1000, demo):
    for d in constant_demo_1, constant_demo_1000, demo:
        ts = np.append(np.cumsum(np.exp(d.logs)), np.inf)
        a = np.exp(d.logu)
        b = (np.array(d.logv) - np.array(d.logu)) / (ts[1:] - ts[:-1])
        def f(t):
            i = np.maximum(np.searchsorted(ts, t) - 1, 0)
            ret = a[i] * np.exp(b[i] * (t - ts[i]))
            assert not np.isnan(ret)
            return ret
        for t in ls:
            I = scipy.integrate.quad(f, 0, t)
            print(I, d.R(t))
            assert(abs(I[0] - d.R(t)) < 1e-6)


