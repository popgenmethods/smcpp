import pytest
import numpy as np
import scipy.integrate

import _pypsmcpp

@pytest.fixture
def constant_demo_1():
    a = np.array([1.0])
    b = np.array([1e-6])
    s = np.array([0.0])
    return (a, b, s)

@pytest.fixture
def constant_demo_1000():
    a = np.array([1. / 1000.])
    b = np.array([1e-6])
    s = np.array([0.0])
    return (a, b, s)
