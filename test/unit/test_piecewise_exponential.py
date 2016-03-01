import pytest
import numpy as np
from scipy.interpolate import PPoly
from scipy.integrate import quad
from piecewise_exponential import PExp

@pytest.fixture
def random_pexp():
    K = 4
    ts = np.array([0.] + sorted(np.random.exponential(2., size=K - 1)))
    # p = PPoly(np.random.normal(1., 2., size=(2, K)), dom)
    pc = np.zeros((1, K))
    qc = np.random.normal(-1, 3, size=(1, K))**2
    b = np.random.normal(size=K)
    # np.random.normal(size=K)
    return PExp(ts, pc, qc, b)

def test_inverse(random_pexp):
    i0 = random_pexp.integral0()
    for x, xinv in zip(i0.ts, i0.inverse(i0.ts)):
        assert abs(x - i0(xinv)) < 1e-8


def test_call():
    ts = np.array([0, 1., 2., 3., 4.])
    qc = [np.exp(ts)]
    pc = [[0, 0, 0, 0, 0]]
    b = [1., 1., 1., 1., 1.]
    pe = PExp(ts, pc, qc, b)
    t = np.linspace(0, 10, 100)
    assert np.allclose(pe(t), np.exp(t))

def test_integral0_simple():
    ts = [0]
    pc = [[0.]]
    qc = [[1.]]
    b = [1]
    pe = PExp(ts, pc, qc, b)
    ipe = pe.integral0()
    assert np.allclose(ipe.ts, pe.ts)
    assert np.allclose(ipe.b, pe.b)
    assert np.allclose(ipe.p.c, [[-1]])
    assert np.allclose(ipe.q.c, pe.q.c)
    for x in (0.0, 0.01, 0.02, 1.0, 10.0):
        assert abs((pe(x) - 1.) - ipe(x)) < 1e-8

def test_random_integral(random_pexp):
    rp_anti = random_pexp.integral0()
    t = np.linspace(0, 10., 100)
    for tt in t:
        q = quad(random_pexp, 0., tt, epsabs=1e-12, 
                points=[t for t in random_pexp.ts if 0 < t < tt])
        v = rp_anti(tt)
        assert abs(q[0] - v) <= 1e-8

def test_infinite_integral(random_pexp):
    # Ensure that limit exists
    random_pexp.b[-1] = -np.abs(random_pexp.b[-1])
    rp_anti = random_pexp.integral0()
    q = quad(random_pexp, 0., np.inf, epsabs=1e-12)
    v = rp_anti.limit_infinity
    if abs(v) == np.inf:
        assert q[0] == np.inf
    else:
        assert abs(q[0] - v) <= 1e-6

def test_limit(random_pexp):
    ts = np.array([0, 1., 2., 3., 4.])
    qc = [np.exp(ts)]
    pc = [[0, 0, 0, 0, 5]]
    b = [1., 1., 1., 1., -1.]
    pe = PExp(ts, pc, qc, b)
    assert abs(pe.limit_infinity - 5.) < 1e-8

def test_bug1():
    array = np.array
    ts = array([ 0. ,  0.1,  0.2,  0.5])
    pc = [[ 0.,  0.,  0.,  0.]]
    qc = [[ 10.5 ,   4.2 ,  37.8 ,  54.6 ], 
          [  0.  ,   0.42,   7.56,  27.3 ]]
    b = array([-10.5, -10.5, -10.5, -10.5])
    pe = PExp(ts, pc, qc, b)
    tt = np.linspace(0, 10, 100)
    anti = pe.integral0()
    for t in tt:
        q = quad(pe, 0., t, epsabs=1e-12, 
                points=[x for x in anti.ts if 0 < x < t])
        v = anti(t)
        assert abs(q[0] - v) <= 1e-6

def test_bug2():
    a = np.array([1.0])
    b = np.array([.000001])
    s = np.array([0.0])
    K = 1
    eta = PExp(np.cumsum(s**2), np.zeros([1, K]), [a**2], b)
    R = eta.integral0()
    assert R(0) == 0.0

def test_derivative_of_integral():
