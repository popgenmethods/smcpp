import pytest
import numpy as np
import scipy.integrate

import _pypsmcpp

@pytest.fixture
def demo():
    u = np.log([1.0, 2.0, 3.0, 4.0])
    v = np.log([2.0, 3.0, 4.0, 5.0])
    s = np.log([0.0, .3, .8, 1.0])
    d = _pypsmcpp.Demography(u, v, s)
    return d

@pytest.fixture
def constant_demo_1():
    u = np.log([1.0])
    v = np.log1p([1e-6])
    s = np.array([0.0])
    d = _pypsmcpp.Demography(u, v, s)
    return d

@pytest.fixture
def constant_demo_1000():
    u = np.log([1000.0])
    v = np.log([1001.0])
    s = np.array([0.0])
    d = _pypsmcpp.Demography(u, v, s)
    return d
