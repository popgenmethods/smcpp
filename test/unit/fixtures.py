import pytest
import numpy as np

import smcpp._smcpp, smcpp.model

@pytest.fixture
def im():
    K = 10
    hidden_states = np.concatenate([[0.], np.logspace(-2, 1, 10), [np.inf]])
    N0 = 10000.
    theta = 1.25e-8
    rho = theta / 4.
    n = 26
    M = len(hidden_states) - 1
    s = np.diff(np.logspace(np.log10(.01), np.log10(3.), 41))
    obs_list = [np.array([[1, 0, 10, n - 2], [10, 0, 0, n - 2], [1, 2, 2, n - 5]], dtype=np.int32)]
    im = smcpp._smcpp.PyInferenceManager(n - 2, obs_list, hidden_states, s)
    im.rho = 4.0 * N0 * rho
    im.theta = 4.0 * N0 * theta
    knots = np.logspace(np.log10(.01), np.log10(3.), K)
    model = smcpp.model.SMCModel(s, knots)
    im.model = model
    return im
