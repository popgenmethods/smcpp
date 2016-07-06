import pytest
import numpy as np

import smcpp
from smcpp.jcsfs import JointCSFS
from smcpp.model import SMCModel

@pytest.fixture
def model1():
    s = np.diff(np.logspace(np.log10(.001), np.log10(4), 20))
    ret = SMCModel(s, np.logspace(np.log10(.001), np.log10(4), 5))
    ret.y[:] = np.log(np.arange(1, 6)[::-1])
    return ret

@pytest.fixture
def model2():
    s = np.diff(np.logspace(np.log10(.001), np.log10(4), 20))
    ret = SMCModel(s, np.logspace(np.log10(.001), np.log10(4), 5))
    ret.y[:] = np.log(np.arange(1, 6))
    return ret

def test_jcsfs(model1, model2):
    jcsfs = JointCSFS(8, 8, 2, 0, model1, model2, 1.0)
    jcsfs._jcsfs_helper_tau_below_split(0.5, 0.8)
