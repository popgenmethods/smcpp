import _pypsmcpp
import numpy as np
np.set_printoptions(linewidth=120)
import sys
from fixtures import *
from _pypsmcpp import Demography

def test_correct_const_1000(constant_demo_1000):
    # Make as close to constant as possible
    for n in (2, 3, 10, 20):
        oaeu
        rsfs, sfs, jac = _pypsmcpp.sfs(constant_demo_1000, 500, 100, n, 0., np.inf, 
                1.0, extract_output=True, numthreads=8)
        print(n)
        print(rsfs)
        for k in range(1, n):
            assert abs((rsfs[k] - 2000. / k)  / (2000. / k)) < 1e-1

def test_correct_const(constant_demo):
    # Make as close to constant as possible
    for n in (2, 3, 10, 20):
        rsfs, sfs, jac = _pypsmcpp.sfs(constant_demo, 500, 100, n, 0., np.inf, 
                1.0, extract_output=True, numthreads=8)
        print(n)
        print(rsfs)
        for k in range(1, n):
            assert abs((rsfs[k] - 2. / k) / (2. / k)) < 1e-1

def test_d():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([.00001, -.001, -.01, .01])
    s = np.array([0.0, .3, .8, 1.0])
    K = s.shape[0]
    M = 100
    S = 500
    n = 15
    demo = Demography(a, b, s)
    rsfs, sfs, jac = _pypsmcpp.sfs(demo, S, M, n, 0., 10, 1.0,
            numthreads=2, extract_output=True, seed=1)
    eps = .02
    I = np.eye(K)
    for ind in (0, 1, 2):
        for k in range(K):
            args = [a, b, s]
            if ind == 2 and k == 0:
                pass
            args[ind] = args[ind] + eps * I[k]
            print(args)
            demo2 = Demography(*args)
            _, sfs2, jac2 = _pypsmcpp.sfs(demo2, S, M, n, 0., 10, 1.0,
                    numthreads=2, extract_output=True, seed=1)
            for i in (0, 1, 2):
                for j in range(n + 1):
                    jaca = jac[i, j, ind, k]
                    j1 = sfs2[i, j]
                    j2 = sfs[i, j] + eps * jaca
                    print(ind, k, i, j, sfs2[i,j], sfs[i,j], jaca, abs(j2 - j1))
                    # assert abs(j1 - j2) < eps

