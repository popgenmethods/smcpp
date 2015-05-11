import pytest
import numpy as np
import demography
import _pypsmcpp

@pytest.fixture
def demo():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([.00001, -.001, -.01, .01])
    s = np.array([0.0, .3, .8, 1.0])
    demo = demography.Demography(a, b, s, [0.0, 1.0, 2.0, 3.0, np.inf])
    return demo

@pytest.fixture
def hs():
    return np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0, 13.0, np.inf])

def test_derivative(demo, hs):
    rho = 1e-9
    mat, jac = _pypsmcpp.transition(demo, hs, rho, True)
    print(mat)
    a = demo.sqrt_a
    b = demo.b
    s = demo.sqrt_s
    M = mat.shape[0]
    eps = .1
    K = demo.K
    I = np.eye(K)
    for ind in (0, 1, 2):
        for k in range(K):
            args = [a, b, s, [0.0, 1.0, 2.0, 3.0, np.inf]]
            if ind == 2 and k == 0:
                pass
            args[ind] = args[ind] + eps * I[k]
            demo2 = demography.Demography(*args)
            mat2, _ = _pypsmcpp.transition(demo2, hs, rho, True)
            for i in range(M):
                for j in range(M):
                    jaca = jac[i, j, ind, k]
                    j1 = mat2[i, j]
                    j2 = mat[i, j] + eps * jaca
                    # print(ind, k, i, j, jaca, j1, j2, j1 - j2)
                    assert abs(j1 - j2) < eps

def test_sum_to_one(demo, hs):
    mat, jac = _pypsmcpp.transition(demo, hs, 1e-8, True)
    assert np.all(np.abs(mat.sum(axis=1) - 1) < 1e-5)
