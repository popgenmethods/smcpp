import numpy as np
np.set_printoptions(linewidth=120)
import sys

import _pypsmcpp
from fixtures import *

NTHREAD = 1
num_samples = 200

def test_d():
    a = np.array([0.1, 0.1, 0.2, 0.3])
    b = np.array([0.2, 0.3, .8, 1.0])
    K = a.shape[0]
    s = np.array([0.3] * K)
    n = 10
    t0 = 0.0
    t1 = np.inf
    sfs, _, jac, rjac = _pypsmcpp.sfs((a, b, s), n, num_samples, t0, t1, NTHREAD, 1.0, jacobian=True, seed=1)
    eps = 2e-8
    I = np.eye(K)
    for ind in (0, 1):
        for k in range(K):
            args = [a, b]
            args[ind] = args[ind] + eps * I[k]
            print(args)
            la, lb = args
            sfs2, rsfs2 = _pypsmcpp.sfs((la, lb, s), n, num_samples, t0, t1, NTHREAD, 1.0, jacobian=False, seed=1)
            for i in (0, 1):
                for j in range(n + 1):
                    jaca = jac[i, j, ind, k]
                    j1 = sfs2[i, j]
                    j2 = sfs[i, j] + eps * jaca
                    print(ind, k, i, j, jaca, (sfs2[i,j] - sfs[i,j]) / eps)
                    # assert abs(j1 - j2) < eps

def test_matching_diff_nodiff(demo):
    from timeit import default_timer as timer
    a, b, s = demo
    n = 5
    start = timer()
    seed = np.random.randint(0, 10000000)
    sfs1, rsfs1, jac = _pypsmcpp.sfs((a, b, s), n, num_samples, 0., np.inf, NTHREAD, 1.0, jacobian=True, seed=seed)
    end = timer()
    print("Run-time with jacobian: %f" % (end - start))
    start = timer()
    sfs2, rsfs2 = _pypsmcpp.sfs((a, b, s), S, M, 0., np.inf, NTHREAD, 1.0, jacobian=False, seed=seed)
    end = timer()
    print("Run-time without: %f" % (end - start))
    print(sfs1)
    print(sfs2)
    print(jac)

def test_correct_const(constant_demo_1, constant_demo_1000):
    # Make as close to constant as possible
    obs = [np.array([[1, 0, 0], [1, 0, 0]], dtype=np.int32)]
    for d, mult in ((constant_demo_1, 1.0), (constant_demo_1000, 1000.0)):
        for n in (2, 3, 10, 20):
            sfs, rsfs, jac = _pypsmcpp.sfs((a, b, s), n, S, M, 0., np.inf, NTHREAD, 1.0, jacobian=True, seed=seed)
            im = _pypsmcpp.PyInferenceManager(n - 2, obs, np.array([0.0, np.inf]), 1e-8, 1e-8, 1, NTHREAD, num_samples)
            sfs = im.sfs(d, 0, np.inf)
            rsfs = _pypsmcpp.reduced_sfs(sfs) 
            print(rsfs)
            rsfs /= rsfs[1:].sum()
            print(rsfs)
            for k in range(1, n):
                expected = (1. / k) / sum([1. / j for j in range(1, n)])
                assert (rsfs[k] - expected) / expected < 1e-2

