import pytest
import ad
import numpy as np

import smcpp
from smcpp import util
from smcpp.jcsfs import JointCSFS
from smcpp.model import SMCModel

@pytest.fixture
def model1():
    s = np.diff(np.logspace(np.log10(.001), np.log10(4), 20))
    ret = SMCModel(s, np.logspace(np.log10(.001), np.log10(4), 5))
    ret[:] = np.log(np.arange(1, 6)[::-1])
    ret[:] = 0.
    return ret

@pytest.fixture
def model2():
    s = np.diff(np.logspace(np.log10(.001), np.log10(4), 20))
    ret = SMCModel(s, np.logspace(np.log10(.001), np.log10(4), 5))
    ret[:] = np.log(np.arange(1, 6))
    ret[:] = 0.
    return ret

@pytest.fixture
def jcsfs():
    return JointCSFS(5, 2, 2, 0, [0.0, 0.5, 1.0, np.inf])

np.set_printoptions(precision=3, linewidth=100)

def _test_d(model1, model2):
    ts = [0.0, 0.5, 1.0, np.inf]
    split = 0.25
    n1 = 10 
    n2 = 8
    j = JointCSFS(n1, n2, 2, 0, ts, 100)
    ders = model1[:3] = ad.adnumber(model1[:3])
    j0 = j.compute(model1, model2, split)
    model1.reset_derivatives()
    for i in range(3):
        model1[i] += 1e-8
        j1 = j.compute(model1, model2, split)
        model1[i] -= 1e-8
        for x, y in zip(j0.flat, j1.flat):
            print(x.d(ders[i]), (y - x) * 1e-8)
    assert False
    

def test_marginal_pop1(model1, model2):
    ts = [1., 2.]
    n1 = 0
    n2 = 10
    j = JointCSFS(n1, n2, 2, 0, ts, 100)
    for split in [0.5]:
        jc = j.compute(model1, model2, split)
        for t1, t2, jjc in zip(ts[:-1], ts[1:], jc):
            A1 = smcpp._smcpp.raw_sfs(model1, n1, t1, t2).astype('float')
            A2 = jjc.sum(axis=(-1, -2)).astype('float')
            A1[0,0] = A2[0,0]
            A1[-1,-1] = A2[-1,-1]
            print((t1, t2, split))
            print(A1)
            print(A2)
            print(((A1 - A2) / A1).round(3))
            # assert np.allclose(A1.flat[1:-1], A2.flat[1:-1])

def _test_marginal_pop2(model1, model2):
    ts = [0.0, 0.5, 1.0, np.inf]
    n1 = 10 
    n2 = 8
    j = JointCSFS(n1, n2, 2, 0, ts, 100)
    for split in [0.1, 0.25, 0.5, 0.75, 1.0, 2.0]:
        jc = j.compute(model1, model2, split)
        for t1, t2, jjc in zip(ts[:-1], ts[1:], jc):
            A1 = util.undistinguished_sfs(smcpp._smcpp.raw_sfs(model2, n2 - 2, 0., np.inf)).astype('float')
            A2 = jjc.sum(axis=(0, 1, 2)).astype('float')
            print((A1.flat[1:-1] - A2.flat[1:-1]) / A2.flat[1:-1])
            # assert np.allclose(A1, A2, 1e-1, 0)

# def test_jcsfs(jcsfs, model1, model2):
#     jcsfs.compute(model1, model2, 0.25)
