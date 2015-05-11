import _pypsmcpp
import numpy as np
np.set_printoptions(linewidth=120)
import sys
from fixtures import *

def test_correct_const(constant_demo):
    # Make as close to constant as possible
    for n in (2, 3, 10, 20):
        rsfs, sfs, jac = _pypsmcpp.sfs(constant_demo, 100, 500, n, 0., np.inf, True)
        print(n)
        print(rsfs)
        for k in range(1, n):
            assert abs((rsfs[k] - 2. / k) / (2. / k)) < 1e-1

def test_d():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([.00001, -.001, -.01, .01])
    s = np.array([0.0, .3, .8, 1.0])
    K = s.shape[0]
    M = 100000
    n = 10
    demo = demography.Demography(a, b, s, [0.0, 1.0, 2.0, 3.0, np.inf])
    rsfs, sfs, jac = _pypsmcpp.sfs(demo, 100, 500, n, 3., 4, extract_output=True)
    eps = .1
    I = np.eye(K)
    for ind in (0, 1, 2):
        for k in range(K):
            args = [a, b, s, [0.0, 1.0, 2.0, 3.0, np.inf]]
            if ind == 2 and k == 0:
                pass
            args[ind] = args[ind] + eps * I[k]
            demo2 = demography.Demography(*args)
            _, sfs2, jac2 = _pypsmcpp.sfs(demo2, 100, 500, n, 3., 4, extract_output=True)
            for i in (0, 1, 2):
                for j in range(n + 1):
                    jaca = jac[i, j, ind, k]
                    j1 = sfs2[i, j]
                    j2 = sfs[i, j] + eps * jaca
                    print(ind, k, i, j, sfs2[i,j], sfs[i,j], jaca, abs(j2 - j1))
                    assert abs(j1 - j2) < eps

