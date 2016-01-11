from __future__ import division
import numpy as np
import _pypsmcpp
import multiprocessing
import lib.scrm as scrm
from lib.util import sawtooth

np.set_printoptions(suppress=True, linewidth=120)
M = 1000
THREADS = 16
theta = 1.25e-8

def _scrm_sfs(args):
    np.random.seed(args[-1])
    return scrm.distinguished_sfs(*(args[:-1]))

def _test_two_period0():
    N0 = 10000
    a, b = np.array([[  9.133866,   6.13109 ,   7.215701,   8.550314,   8.66131 ,   8.816327,   9.011795,   9.260647,   9.625925,  10.059977,  10.586635,  11.215727,  11.984432,  12.980205,  14.221971,  15.671126,
        20.      ,  19.185278,  20.      ,  20.      ,  20.      ,  20.      ,   0.535405,   0.677208,   0.920769,   0.814968,   0.592878,  13.87953 ,   0.439724,   0.10269 ],
        [  9.278037,   5.882157,   6.844227,   8.070035,   8.813656,   8.967176,   9.15853 ,   9.421955,   9.773194,  10.199097,  10.711501,  11.326015,  12.097228,  13.105783,  14.313083,  15.773711,
            20.      ,  19.39545 ,  20.      ,  20.      ,  20.      ,  20.      ,   0.628173,   0.472427,   0.848701,   0.983914,   0.579022,   5.697205,   0.486631,   1.2     ]])
    s = np.array([ 0.002   ,  0.000546,  0.000695,  0.000885,  0.001127,  0.001435,  0.001827,  0.002326,  0.002962,  0.003771,  0.004801,  0.006112,  0.007781,  0.009907,  0.012612,  0.016057,  0.020443,
                0.026027,  0.033136,  0.042187,  0.053709,  0.068379,  0.087056,  0.110835,  0.141108,  0.17965 ,  0.22872 ,  0.291192,  0.370727,  0.471987])
    n = 10
    L = 100000
    t0, t1 = 0., np.inf
    sfs = _pypsmcpp.sfs(n, [a, b, s], t0, t1, 2 * N0 * theta, jacobian=False)
    # Everything in ms is in units of 4 * N0
    print(sfs)
    demography = scrm.demography_from_params(np.array([a, b, s]))
    print(demography)
    args = (n, L, N0, theta, demography, t0, t1)
    scrm_sfs = np.mean(list(multiprocessing.Pool(THREADS).map(_scrm_sfs, [args + (np.random.randint(0, 1000000),) for _ in range(THREADS)])), axis=0)
    # scrm_sfs = scrm.distinguished_sfs(*args)
    print(sfs)
    print("")
    print(scrm_sfs)
    print("")
    # assert False

def test_two_period1():
    N0 = 10000.
    # a = np.array([8.0, 8.0, .75, .75, 4.0, 4.0])
    # b = a + .5
    # s = np.array([0.5] * 6) * 2.0
    a = np.array([15.0, 1.0, 2.0])
    b = np.array([1.0, 1.0, 2.0])
    s = np.array([2.0, 1.0, 1.0])
    n = 10
    L = 100000
    t0, t1 = 0.0, 14.9
    sfs = _pypsmcpp.sfs(n, [a, b, s], t0, t1, 2 * N0 * theta, jacobian=False)
    # Everything in ms is in units of 4 * N0
    print(sfs)
    demography = scrm.demography_from_params(np.array([a, b, s * 0.5]))
    print(demography)
    args = (n, L, N0, theta, demography, t0, t1)
    scrm_sfs = np.mean(list(multiprocessing.Pool(THREADS).map(_scrm_sfs, [args + (np.random.randint(0, 1000000),) for _ in range(THREADS)])), axis=0)
    # scrm_sfs = scrm.distinguished_sfs(*args)
    print(sfs)
    print("")
    print(scrm_sfs)
    print("")
    # assert False

def _test_two_period():
    a = np.array([1., 20., 0.01])
    b = np.array([0.01, -0.01, 0.0])
    s = np.array([0.15, 0.15, 0.2]) * 2
    tb = np.append(a[:-1] * np.exp(b[:-1] * s[:-1]), 0)
    n = 5
    N0 = 10000
    theta = 1e-8
    sfs, rsfs = _pypsmcpp.sfs([a, tb, s], n - 2, M, 0., np.inf, THREADS, 4 * N0 * theta / 2.0, jacobian=False)
    L = 1000000
    args = (n, L, N0, theta, ['-eG', 0.0, .01, '-eN', 0.15, 1. / 20., '-eG', 0.15, -0.01, '-eN', 0.3, 100.0]) # , '-eN', 0.25, 4.])
    scrm_sfs = np.mean(list(multiprocessing.Pool(THREADS).map(_scrm_sfs, [args for _ in range(THREADS)])), axis=0) / L
    print("")
    print(rsfs)
    print(scrm_sfs)
    assert False

def _test_two_period_flat():
    a = np.log([1., 2.0])
    b = np.array([0.00, 0.0])
    s = np.array([0.2, 0.1])
    ts = np.cumsum(s)
    tb = np.append(a[:-1] + b[:-1] * (ts[1:] - ts[:-1]), 0)
    n = 10
    N0 = 10000
    theta = 1e-6
    sfs, rsfs = _pypsmcpp.sfs([a, tb, s], n - 2, M, 0., np.inf, THREADS, 4 * N0 * theta / 2.0, jacobian=False)
    args = (n, 100000, N0, theta, ['-eN', 0.1, 2.0])
    scrm_sfs = np.mean(list(multiprocessing.Pool(THREADS).map(_scrm_sfs, [args for _ in range(THREADS)])), axis=0)
    print("")
    print(rsfs)
    print(scrm_sfs)

def _test_one_period():
    a = np.log([1.])
    b = np.array([0.01])
    s = np.array([0.1])
    ts = np.cumsum(s)
    tb = np.append(a[:-1] + b[:-1] * (ts[1:] - ts[:-1]), 0)
    n = 10
    N0 = 10000
    theta = 1e-8
    sfs, rsfs = _pypsmcpp.sfs([a, tb, s], n - 2, M, 0., np.inf, THREADS, 4 * N0 * theta / 2.0, jacobian=False)
    args = (n, 100000, N0, theta, [])
    scrm_sfs = np.mean(list(multiprocessing.Pool(THREADS).map(_scrm_sfs, [args for _ in range(THREADS)])), axis=0)
    print("")
    print(rsfs)
    print(scrm_sfs)
