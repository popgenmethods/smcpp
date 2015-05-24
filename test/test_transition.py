import pytest
import numpy as np

import _pypsmcpp
from fixtures import *

rho = 1e-4

@pytest.fixture
def hs():
    return np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0, 13.0, np.inf])

def test_equal_jac_nojac(constant_demo_1, hs):
    from timeit import default_timer as timer
    a, b, s = constant_demo_1
    start = timer()
    trans1, jac = _pypsmcpp.transition(a, b, s, hs, rho, True)
    end = timer()
    print("Run-time with jacobian: %f" % (end - start))
    start = timer()
    trans2 = _pypsmcpp.transition(a, b, s, hs, rho, False)
    end = timer()
    print("Run-time without: %f" % (end - start))
    assert np.allclose(trans1 - trans2, 0)

def test_sum_to_one(constant_demo_1, hs):
    a, b, s = constant_demo_1
    trans1 = _pypsmcpp.transition(a, b, s, hs, rho, False)
    assert np.allclose(np.sum(trans1, axis=1), 1.0)

def test_d(constant_demo_1, hs):
    a, b, s = constant_demo_1
    K = len(a)
    eps = 0.02
    trans1, jac = _pypsmcpp.transition(a, b, s, hs, rho, True)
    I = np.eye(K)
    M = trans1.shape[0]
    for ind in (0, 1, 2):
        for k in range(K):
            args = [a, b, s]
            if ind == 2 and k == 0:
                pass
            args[ind] = args[ind] + eps * I[k]
            ap, bp, sp = args
            trans2 = _pypsmcpp.transition(ap, bp, sp, hs, rho, False)
            for i in range(M):
                for j in range(M):
                    jaca = jac[i, j, ind, k]
                    j1 = trans2[i, j]
                    j2 = trans1[i, j] + eps * jaca
                    print(ind, k, i, j, trans2[i,j], trans1[i,j], jaca, abs(j2 - j1))
                    # assert abs(j1 - j2) < eps
