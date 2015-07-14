from __future__ import division
import numpy as np
import _pypsmcpp
import multiprocessing
import scrm

np.set_printoptions(suppress=True, linewidth=120)
M = 1000
THREADS = 16
theta = 1e-8

def _scrm_sfs(args):
    return scrm.distinguished_sfs(*args)

def test_two_period0():
    N0 = 10000
    a = np.array([10., 1., .1, 2.])
    b = np.array([1., 1., .1, 2.])
    s = np.array([5000.0, 50000.0, 10000., 10000.]) / 25.0 / 2 * N0
    n = 7
    L = 200000
    # Everything in ms is in units of 4 * N0
    demography = scrm.demography_from_params(np.array([a * 2.0, b * 2.0, s]))
    print(demography)
    t0, t1 = 5.108226,   6.931416
    args = (n, L, N0, theta, demography, t0, t1)
    scrm_sfs = np.mean(list(multiprocessing.Pool(THREADS).map(_scrm_sfs, [args for _ in range(THREADS)])), axis=0)
    # scrm_sfs = scrm.distinguished_sfs(*args, t0=1.0, t1=2.0)
    print("")
    print(scrm_sfs)
    sfs, rsfs = _pypsmcpp.sfs([a, b, s], n - 2, M, t0, t1, THREADS, 4 * N0 * theta, jacobian=False)
    print(sfs)
    print("")
    print(_pypsmcpp.reduced_sfs(scrm_sfs))
    print(rsfs)
    assert False

def test_two_period1():
    a = np.log([8.0, 8.0, .75, .75, 4.0, 4.0])
    b = a + .5
    s = np.array([0.5] * 6) * 2.0
    n = 10
    N0 = 10000
    sfs, rsfs = _pypsmcpp.sfs([a, b, s], n - 2, M, 0., np.inf, THREADS, 4 * N0 * theta / 2.0, jacobian=False)
    L = 100000
    demography = scrm.demography_from_params([a, b, s])
    print(demography)
    args = (n, L, N0, theta, demography)
    scrm_sfs = np.mean(list(multiprocessing.Pool(THREADS).map(_scrm_sfs, [args for _ in range(THREADS)])), axis=0) / L
    print("")
    print(rsfs)
    print(np.abs(rsfs - scrm_sfs))
    assert False

def test_two_period():
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

def test_two_period_flat():
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

def test_one_period():
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
