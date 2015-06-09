import pytest
import numpy as np
import scipy.integrate

import _pypsmcpp

@pytest.fixture
def fake_obs():
    ary = []
    for ell in range(L):
        ary.append([np.random.randint(1, 1000), 0, 0])
        d = np.random.randint(0, 3)
        ary.append([1, d, np.random.randint(not d, N + 1 - (d == 2))])
    return np.array(ary)

@pytest.fixture
def constant_demo_1():
    a = np.log([1.0])
    b = np.log([1.0])
    s = np.array([1.0])
    return (a, b, s)

@pytest.fixture
def constant_demo_1000():
    a = np.log([1000.0])
    b = np.log([1000.0])
    s = np.array([1.0])
    return (a, b, s)

@pytest.fixture
def demo():
    a = np.array([.2, 1., 2., 3.])
    b = np.array([.3, 2., 1., 3.5])
    s = np.array([0.1, 0.2, 0.3, 0.4])
    return (a, b, s)
