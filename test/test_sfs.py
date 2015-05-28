import numpy as np
np.set_printoptions(linewidth=120)
import sys

import _pypsmcpp
from fixtures import *

NTHREAD = 32
S = 2000
M = 100

def test_d():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    sqrt_y = np.sqrt([0.5, .3, .8, 1.0])
    K = x.shape[0]
    S = 1000
    M = 50
    n = 5
    sfs, _, jac = _pypsmcpp.sfs(x, sqrt_y, n, S, M, 0., np.inf, NTHREAD, 1.0, jacobian=True, seed=1)
    eps = .2
    I = np.eye(K)
    for ind in (0, 1):
        for k in range(K):
            args = [x, sqrt_y]
            args[ind] = args[ind] + eps * I[k]
            print(args)
            xp, sqrt_yp = args
            sfs2, rsfs2 = _pypsmcpp.sfs(xp, sqrt_yp, n, S, M, 0., np.inf, NTHREAD, 1.0, jacobian=False, seed=1)
            for i in (0, 1):
                for j in range(n + 1):
                    jaca = jac[i, j, ind, k]
                    j1 = sfs2[i, j]
                    j2 = sfs[i, j] + eps * jaca
                    print(ind, k, i, j, sfs2[i,j], sfs[i,j], jaca, abs(j2 - j1))
                    # assert abs(j1 - j2) < eps

def test_correct_const(constant_demo_1, constant_demo_1000):
    # Make as close to constant as possible
    for d, mult in ((constant_demo_1, 1.0), (constant_demo_1000, 1000.0)):
        x, sqrt_y = d
        for n in (2, 3, 10, 20):
            sfs, rsfs = _pypsmcpp.sfs(x, sqrt_y, n, S, M, 0., np.inf, NTHREAD, 1.0, jacobian=False)
            print(n)
            print(rsfs)
            for k in range(1, n):
                assert abs((rsfs[k] - 2. * mult / k) / (2. * mult / k)) < 2e-2

def test_matching_diff_nodiff(constant_demo_1):
    from timeit import default_timer as timer
    x, sqrt_y = constant_demo_1
    n = 5
    start = timer()
    seed = np.random.randint(0, 10000000)
    sfs1, rsfs1, jac = _pypsmcpp.sfs(x, sqrt_y, n, S, M, 0., np.inf, NTHREAD, 1.0, jacobian=True, seed=seed)
    end = timer()
    print("Run-time with jacobian: %f" % (end - start))
    start = timer()
    sfs2, rsfs2 = _pypsmcpp.sfs(x, sqrt_y, n, S, M, 0., np.inf, NTHREAD, 1.0, jacobian=False, seed=seed)
    end = timer()
    print("Run-time without: %f" % (end - start))
    print(sfs1)
    print(sfs2)
    print(jac)

