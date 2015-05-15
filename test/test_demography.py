import numpy as np
from multiprocessing import Pool
import pytest

# import warnings
# warnings.filterwarnings("error")

from _pypsmcpp import Demography

def test_bug1():
    sqrt_a = np.array([ 0.0316228 ,  0.03162278,  0.04361865])
    b = np.array([  1.00000086e-04,   1.00000000e-04,  -3.22445023e-01])
    sqrt_s = np.array([ 0.        ,  0.70710678,  0.99999996])
    demo = Demography(sqrt_a, b, sqrt_s)
    assert not np.isnan(demo.inverse_rate(1.25))

@pytest.fixture
def constant_demo():
    return Demography([1.0], [.000001], [0.0], [0.0, 0.5, 1.0, 2.0, 3.0, np.inf])

@pytest.fixture
def nonconstant_demo():
    return Demography([1.0, 0.5, 3.0], [.01, -0.01, 0.1], [0.0, .1, 0.3], [0.0, 0.5, 1.0, 2.0, 3.0, np.inf])

def test_pi(constant_demo):
    pi, jac = constant_demo.pi()
    for prb, bds in zip(pi, zip(constant_demo.hs[:-1], constant_demo.hs[1:])):
        p2 = np.exp(-np.array(bds))
        assert abs(prb - (p2[0] - p2[1])) < 1e-6

def test_pi_sum_to_one(constant_demo, nonconstant_demo):
    for d in constant_demo, nonconstant_demo:
        pi, _ = d.pi()
        assert abs(pi.sum() - 1.0) < 1e-6

def test_pi_da(nonconstant_demo):
    cd = nonconstant_demo
    K = cd.sqrt_a.shape[0]
    pi, jac = cd.pi()
    eps = 1e-2
    I = np.eye(K)
    for k in range(K):
        d2 = Demography(cd.sqrt_a + eps * I[k], cd.b, cd.sqrt_s, cd.hs)
        pi2, _ = d2.pi()
        # print(k)
        # print(jac[0, k])
        # print(pi)
        # print(pi2)
        assert np.max(abs(pi2 - (pi + eps * jac[0, k]))) < eps

def test_pi_db(nonconstant_demo):
    cd = nonconstant_demo
    K = cd.sqrt_a.shape[0]
    pi, jac = cd.pi()
    eps = 1e-3
    I = np.eye(K)
    for k in range(K):
        d2 = Demography(cd.sqrt_a, cd.b + eps * I[k], cd.sqrt_s, cd.hs)
        pi2, _ = d2.pi()
        # print(k)
        # print(jac[1, k])
        # print(pi)
        assert np.max(abs(pi2 - (pi + eps * jac[1, k]))) < eps

def test_pi_ds(nonconstant_demo):
    cd = nonconstant_demo
    K = cd.sqrt_a.shape[0]
    pi, jac = cd.pi()
    eps = 1e-3
    I = np.eye(K)
    for k in range(1, K):
        d2 = Demography(cd.sqrt_a, cd.b, cd.sqrt_s + eps * I[k], cd.hs)
        pi2, _ = d2.pi()
        # print(k)
        # print(jac[2, k])
        # print(pi)
        assert np.max(abs(pi2 - (pi + eps * jac[2, k]))) < eps

def test_transition_nonconstant(nonconstant_demo):
    A, jac_A = nonconstant_demo.transition(1e-7)

def test_transition(constant_demo):
    A, jac_A = constant_demo.transition(1e-7)

def test_sfs_multiple(constant_demo):
    p = Pool(12)
    constant_demo.pool = p
    constant_demo.sfs(50, 100000, 1e-7)
    p.terminate()

def test_sfs_single(constant_demo):
    constant_demo.sfs(50, 10000, 1e-7)
