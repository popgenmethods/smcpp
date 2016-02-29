import pytest
import numpy as np
import cPickle
import scipy.integrate

import _pypsmcpp

@pytest.fixture
def mock_dataset():
    return cPickle.load(open("test/test_dataset.dat", "rb"))

@pytest.fixture
def im():
    n = 50
    N0 = 10000.
    rho = 1e-9
    theta = 2.5e-8
    fake_obs = np.load("test/obs_list.npy")
    obs_list = [fake_obs[:10]]
    em = np.arange(3 *  (n - 1), dtype=int).reshape([3, n - 1])
    em[0] = em[2] = 0
    em[1] = 1
    im = _pypsmcpp.PyInferenceManager(n - 2, obs_list, np.array([0.0, 0.2, .05, 1.0, 2.0, 3.0, 10.0]),
            4.0 * N0 * theta, 4.0 * N0 * rho, 50, n, [0], em)

@pytest.fixture
def constant_demo_1():
    a = np.array([1.0])
    b = np.array([1.0])
    s = np.array([1.0])
    return (a, b, s)

@pytest.fixture
def constant_demo_1000():
    a = 1. / np.array([1000.0])
    b = 1. / np.array([1000.0])
    s = np.array([1.0])
    return (a, b, s)

@pytest.fixture
def demo():
    a = np.array([.2, 1., 2., 3.])
    b = np.array([.3, 2., 1., 3.5])
    s = np.array([0.1, 0.2, 0.3, 0.4])
    return (a, b, s)

@pytest.fixture
def fake_obs():
    L = 10000
    n = 25
    ary = []
    for ell in range(L):
        ary.append([np.random.randint(1, 1000), 0, 0])
        d = np.random.randint(0, 3)
        ary.append([1, d, np.random.randint(not d, n + 1 - (d == 2))])
    return np.array(ary, dtype=np.int32)

