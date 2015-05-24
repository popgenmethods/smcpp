import numpy as np
np.set_printoptions(linewidth=120)
import sys

import _pypsmcpp
from fixtures import *

NTHREAD = 16
S = 2000
M = 100

def test_matching_diff_nodiff(constant_demo_1):
    from timeit import default_timer as timer
    a, b, s = constant_demo_1
    n = 5
    start = timer()
    seed = np.random.randint(0, 10000000)
    sfs1, rsfs1, jac = _pypsmcpp.sfs(a, b, s, n, S, M, 0., np.inf, NTHREAD, 1.0, jacobian=True, seed=seed)
    end = timer()
    print("Run-time with jacobian: %f" % (end - start))
    start = timer()
    sfs2, rsfs2 = _pypsmcpp.sfs(a, b, s, n, S, M, 0., np.inf, NTHREAD, 1.0, jacobian=False, seed=seed)
    end = timer()
    print("Run-time without: %f" % (end - start))
    print(sfs1)
    print(sfs2)

def test_correct_const(constant_demo_1, constant_demo_1000):
    # Make as close to constant as possible
    for d, mult in ((constant_demo_1, 1.0), (constant_demo_1000, 1000.0)):
        a, b, s = d
        for n in (2, 3, 10, 20):
            sfs, rsfs = _pypsmcpp.sfs(a, b, s, n, S, M, 0., np.inf, NTHREAD, 1.0, jacobian=False)
            print(n)
            print(rsfs)
            for k in range(1, n):
                assert abs((rsfs[k] - 2. * mult / k) / (2. * mult / k)) < 2e-2

def test_d():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([.00001, -.001, -.01, .01])
    s = np.array([0.0, .3, .8, 1.0])
    K = s.shape[0]
    M = 100
    S = 500
    n = 10
    sfs, _, jac = _pypsmcpp.sfs(a, b, s, n, S, M, 0., np.inf, NTHREAD, 1.0, jacobian=True, seed=1)
    eps = .02
    I = np.eye(K)
    for ind in (0, 1, 2):
        for k in range(K):
            args = [a, b, s]
            if ind == 2 and k == 0:
                pass
            args[ind] = args[ind] + eps * I[k]
            print(args)
            ap, bp, sp = args
            sfs2, rsfs2 = _pypsmcpp.sfs(ap, bp, sp, n, S, M, 0., np.inf, NTHREAD, 1.0, jacobian=False, seed=1)
            for i in (0, 1, 2):
                for j in range(n + 1):
                    jaca = jac[i, j, ind, k]
                    j1 = sfs2[i, j]
                    j2 = sfs[i, j] + eps * jaca
                    # print(ind, k, i, j, sfs2[i,j], sfs[i,j], jaca, abs(j2 - j1))
                    assert abs(j1 - j2) < eps

