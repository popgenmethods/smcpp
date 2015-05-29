import numpy as np
import _pypsmcpp
import multiprocessing
import scrm

def _scrm_sfs(args):
    return scrm.sfs(*args)

def test_two_period():
    a = np.array([1., 2., 0.2])
    b = np.array([0.01, -0.01, 0.0])
    s = np.array([0.0, 0.2, 0.4])
    ts = np.cumsum(s)
    tb = np.append(a[:-1] * np.exp(b[:-1] * (ts[1:] - ts[:-1])), 0)
    n = 10
    N0 = 10000
    theta = 1e-8
    sfs, rsfs = _pypsmcpp.sfs([a, tb, s], n - 2, 1000, 100, 0., np.inf, 1, 4 * N0 * theta / 2.0, jacobian=False)
    args = (n, 1000000, N0, theta, ['-eG', 0.0, .01, '-eN', 0.1, 1. / 2., '-eG', 0.1, -0.01, '-eN', 0.3, 5.0]) # , '-eN', 0.25, 4.])
    scrm_sfs = np.mean(list(multiprocessing.Pool(16).map(_scrm_sfs, [args for _ in range(16)])), axis=0)
    print("")
    print(rsfs[1:])
    print(scrm_sfs)

def test_one_period():
    a = np.array([1.0])
    b = np.array([0.0])
    s = np.array([0.0])
    n = 10
    N0 = 10000
    theta = 1e-8
    rsfs, _, = _pypsmcpp.sfs([a, b, s], 1000, 100, n - 2, 0., np.inf, 16, 4 * N0 * theta / 2.0, jacobian=False)
    scrm_sfs = np.mean(list(multiprocessing.Pool(16).map(_scrm_sfs, [(n, 1000000, N0, theta) for _ in range(16)])), axis=0)
    print("")
    print(rsfs[1:])
    print(scrm_sfs)
