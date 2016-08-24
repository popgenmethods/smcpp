from __future__ import division
import numpy as np
import multiprocessing as mp
import logging

import util.scrm as scrm
from smcpp.util import sawtooth, human
from smcpp.model import PiecewiseModel
from smcpp import _smcpp

np.set_printoptions(suppress=False, linewidth=120)

theta = 1.25e-8

def _scrm_sfs(args):
    np.random.seed(args[-1])
    return scrm.sfs(*(args[:-1]))

def test_human():
    # Diploid effective population size
    N0 = 10000.
    a = human['a']
    a[0] = 10.
    b = a
    s = human['s']
    n = 5
    L = 1000000
    t0, t1 = 1.0, 3.0
    model = PiecewiseModel(a, s)
    sfs = _smcpp.raw_sfs(model, n, t0, t1)
    sfs *= 2 * N0 * theta
    sfs[0, 0] = 1. - sfs.sum()
    # print(sfs)
    demography = scrm.demography_from_params(np.array([a, b, s * 0.5]))
    # print(demography)
    args = (n, L, N0, theta, demography, t0, t1)
    # scrm_sfs = np.mean(list(mp.Pool().map(_scrm_sfs, [
    #                    args + (np.random.randint(0, 10000),) for _ in range(32)])), axis=0)
    scrm_sfs = scrm.distinguished_sfs(*args)
    for s in [sfs, scrm_sfs]:
        print(np.array2string(s.astype('float'), formatter={'float_kind': lambda x: "%.3e" % x}))

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
