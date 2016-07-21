import pytest
import numpy as np

import smcpp
from smcpp.jcsfs import JointCSFS
from smcpp.model import SMCModel

@pytest.fixture
def model1():
    s = np.diff(np.logspace(np.log10(.001), np.log10(4), 20))
    ret = SMCModel(s, np.logspace(np.log10(.001), np.log10(4), 5))
    ret[:] = np.log(np.arange(1, 6)[::-1])
    return ret

@pytest.fixture
def model2():
    s = np.diff(np.logspace(np.log10(.001), np.log10(4), 20))
    ret = SMCModel(s, np.logspace(np.log10(.001), np.log10(4), 5))
    ret[:] = np.log(np.arange(1, 6))
    return ret

@pytest.fixture
def jcsfs():
    return JointCSFS(2, 2, 2, 0, [0.0, 0.5, 1.0, np.inf])

def test_marginal(model1, model2):
    j = JointCSFS(2, 0, 2, 0, [0.0, 0.5, 1.0, np.inf])
    jc = j.compute(model1, model2, 0.25)[1]
    cs = smcpp._smcpp.raw_sfs(model1, 2, 0.5, 1.0) 
    print()
    print(jc.shape)
    print(jc.sum(axis=(-1, -2)))
    print(cs)

def test_jcsfs(jcsfs, model1, model2):
    jcsfs.compute(model1, model2, 0.25)
