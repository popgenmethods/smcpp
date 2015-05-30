import pytest
import numpy as np
import scipy.integrate

import _pypsmcpp

@pytest.fixture
def constant_demo_1():
    a = np.array([1.0])
    b = np.array([1.0])
    s = np.array([1.0])
    return (a, b, s)

@pytest.fixture
def constant_demo_1000():
    a, b, s = constant_demo_1
    return (a / 1000., b / 1000., s)

@pytest.fixture
def demo():
    a = np.array([.2, 1., 2., 3.])
    b = np.array([.3, 2., 1., 3.5])
    s = np.array([0.1, 0.2, 0.3, 0.4])
    return (a, b, s)
