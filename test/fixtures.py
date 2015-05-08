import pytest
import numpy as np
import demography

@pytest.fixture
def demo():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([.00001, -.001, -.01, .01])
    s = np.array([0.0, .3, .8, 1.0])
    d = demography.Demography(a, b, s, [0.0, 0.1, 0.2, 0.3, 1.0, np.inf])
    return d

@pytest.fixture
def constant_demo():
    a = np.array([1.0])
    b = np.array([.0000001])
    s = np.array([0.0])
    d = demography.Demography(a, b, s, [0.0, np.inf])
    return d

