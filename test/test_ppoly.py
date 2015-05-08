import pytest
import numpy as np
import mpmath as mp
import pkg_resources
pkg_resources.require("scipy>0.15")
import scipy.interpolate

from piecewise_exponential import PPoly

@pytest.fixture
def polys():
    M = 5
    K = 10
    c = np.random.normal(0., 2., size=(M, K))
    x = sorted(np.array([0] + np.abs(np.random.normal(0., 100., size=(K - 1))).tolist()))
    return (PPoly(c, x), scipy.interpolate.PPoly(c, np.append(x, np.inf)))

def test_call(polys):
    pp, ivp = polys
    for x in np.linspace(0., pp.x[-2] + 10., 100):
        assert abs(pp(x) - ivp(x)) < 1e-6
    for x in pp.x[:-1]:
        assert pp(x) == ivp(x)

def test_derivative(polys):
    pp, ivp = [x.derivative() for x in polys]
    assert np.allclose(pp.x, ivp.x[:-1])
    assert pp.c.shape == ivp.c.shape
    for x in np.linspace(0., pp.x[-2] + 10., 100):
        assert abs(pp(x) - ivp(x)) < 1e-6

def test_antiderivative(polys):
    pp, ivp = [x.antiderivative() for x in polys]
    assert np.allclose(pp.x, ivp.x[:-1])
    assert pp.c.shape == ivp.c.shape
    for x in np.linspace(0., pp.x[-2] + 10., 100):
        assert abs(pp(x) - ivp(x)) < 1e-6
