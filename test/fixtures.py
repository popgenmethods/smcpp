import pytest
import numpy as np
import scipy.integrate

import _pypsmcpp

@pytest.fixture
def constant_demo_1():
    x = np.array([0.0])
    sqrt_y = np.array([1.0])
    return (x, sqrt_y)

@pytest.fixture
def constant_demo_1000():
    x = np.array([0.0])
    sqrt_y = 1. / np.sqrt([1000.])
    return (x, sqrt_y)

@pytest.fixture
def demo():
    x = np.array([0., 1., 2., 3.])
    y = np.array([1., 0.1, 2., 0.3])**2
    return (x, y)
