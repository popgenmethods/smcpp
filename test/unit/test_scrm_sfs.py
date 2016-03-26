from __future__ import division
import numpy as np
import multiprocessing
from util import scrm
from smcpp import _smcpp, model
from smcpp.util import sawtooth, human
import logging
logging.basicConfig(level=logging.DEBUG)

np.set_printoptions(suppress=False, linewidth=120)

M = 1000
THREADS = 16
theta = 1.25e-8

def _scrm_sfs(args):
    np.random.seed(args[-1])
    return scrm.distinguished_sfs(*(args[:-1]))

def test_human():
    # Diploid effective population size
    N0 = 10000.
    # a = np.array([8.0, 8.0, .75, .75, 4.0, 4.0])
    # b = a + .5
    # s = np.array([0.5] * 6) * 2.0
    a = human['a']
    a[0] = 4.
    b = human['b']
    b[0] = 2.
    s = human['s']
    n = 20
    L = 100000
    t0, t1 = 0.0, 48.0
    m = model.SMCModel(s, [0])
    m.x[:2] = [a, b]
    sfs = _smcpp.raw_sfs(m, n, t0, t1, jacobian=False)
    print(sfs)
    aoeu
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
    sfs, rsfs = _psmcpp.sfs([a, tb, s], n - 2, M, 0., np.inf, THREADS, 4 * N0 * theta / 2.0, jacobian=False)
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
    sfs, rsfs = _smcpp.sfs([a, tb, s], n - 2, M, 0., np.inf, THREADS, 4 * N0 * theta / 2.0, jacobian=False)
    args = (n, 100000, N0, theta, [])
    scrm_sfs = np.mean(list(multiprocessing.Pool(THREADS).map(_scrm_sfs, [args for _ in range(THREADS)])), axis=0)
    print("")
    print(rsfs)
    print(scrm_sfs)
