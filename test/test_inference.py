import pytest
import multiprocessing
import numpy as np
import logging

logging.getLogger().setLevel("INFO")

from demography import Demography
import inference

N = 10
L = 1000
hidden_states = np.array([0.0, 1.0, 2.0, np.inf])

@pytest.fixture
def demo():
    return Demography([1.0, 2.0], [.0001, -.01], [0.0, 0.5], [0.0, 0.5, 1.0, 2.0, 3.0, np.inf])

@pytest.fixture
def fake_obs():
    dist = np.random.randint(0, 3, L)
    r2 = np.random.randint(0, N, L)
    ret = np.vstack([dist, np.maximum(1, np.minimum(N - 1, dist - r2))])
    return ret.T

def test_inference(demo, fake_obs):
    logp, jac = inference.log_p(demo, 500, 100, N, fake_obs, 1e-8, 1e-8, hidden_states)
    print(logp)
    print(jac)
    # Well, that worked

def test_derivatives(demo, fake_obs):
    logp, jac = inference.log_p(demo, 200, 1000, N, fake_obs, 1e-8, 1e-8, hidden_states)
    print(jac)
    eps = 0.02
    I = np.eye(demo.K)
    for k in [0, 1, 2]:
        for ell in range(demo.K):
            if k == 2 and ell == 0:
                continue
            args = [demo.sqrt_a.copy(), demo.b.copy(), demo.sqrt_s.copy(), demo.hs]
            args[k][ell] += eps
            demo2 = Demography(*args)
            logp2, _ = inference.log_p(demo2, 200, 1000, N, fake_obs, 1e-8, 1e-8, hidden_states)
            print(k, ell, logp2, logp, eps, logp2 - (logp + eps * jac[k, ell]))
