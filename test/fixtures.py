import pytest
import numpy as np
import _pypsmcpp

@pytest.fixture
def demo():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([.00001, -.001, -.01, .01])
    s = np.array([0.0, .3, .8, 1.0])
    d = _pypsmcpp.Demography(a, b, s)
    return d

@pytest.fixture
def constant_demo_1000():
    a = np.array([1000.0])
    b = np.array([.0000001])
    s = np.array([0.0])
    d = _pypsmcpp.Demography(a, b, s)
    return d

@pytest.fixture
def constant_demo():
    a = np.array([1.0])
    b = np.array([.0000001])
    s = np.array([0.0])
    d = _pypsmcpp.Demography(a, b, s)
    return d

