import pytest
import numpy as np

import _pypsmcpp
import scrm
from fixtures import *

np.set_printoptions(suppress=True)

n = 4
N0 = 10000
theta = 1e-8
rho = 1e-8
L = 20000000
demography = []


@pytest.fixture
def hs():
    return np.array([0.0, 1.0, 1.2, np.inf])

def test_empirical_transition(constant_demo_1, hs):
    data = scrm.simulate(2, N0, theta, rho, L, include_trees=True) # no demography   
    M = hs.shape[0] - 1
    emptrans = np.zeros([M, M])
    fs = frozenset([1, 2])
    first = True
    for span, trans in data[3]:
        t = trans[fs]
        i = np.searchsorted(hs, t) - 1
        if first:
            i_prev = i
            first = False
        assert 0 <= i < M
        emptrans[i_prev, i] += 1
        emptrans[i, i] += span - 1
        i_prev = i
    emptrans /= emptrans.sum(axis=1)[:, None]
    print("transitions:")
    print(emptrans)
    a, b, s = constant_demo_1
    print("transitions2:")
    trans1, jac = _pypsmcpp.transition(constant_demo_1, hs, 4 * N0 * rho, True)
    print(trans1)
