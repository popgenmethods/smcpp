import pytest
import multiprocessing
import numpy as np
import logging

logging.getLogger().setLevel("INFO")

import fpectl
fpectl.turnon_sigfpe()

from demography import Demography
import inference

N = 50
L = 100

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
    prb = inference.log_p(demo, N, 100, fake_obs, 1e-7, 1e-8)
    # Well, that worked

def test_ds(demo, fake_obs):
    p = multiprocessing.Pool(16)
    logp, jac = inference.log_p(demo, N, 100000, fake_obs, 1e-7, 1e-8, p)
    eps = np.array([0, .02])
    demo2 = Demography(demo.sqrt_a, demo.b, demo.sqrt_s + eps, demo.hs)
    logp2, _ = inference.log_p(demo2, N, 100000, fake_obs, 1e-7, 1e-8, p)
    print(logp2, logp, jac, logp2 - (logp + eps[1] * jac[2, 1]))

def test_da(demo, fake_obs):
    p = multiprocessing.Pool(16)
    logp, jac = inference.log_p(demo, N, 100000, fake_obs, 1e-7, 1e-8, p)
    eps = .02
    demo2 = Demography(demo.sqrt_a + eps, demo.b, demo.sqrt_s, demo.hs)
    logp2, _ = inference.log_p(demo2, N, 100000, fake_obs, 1e-7, 1e-8, p)
    print(logp2, logp, jac, logp2 - (logp + eps * jac[0, 0]))

def test_db(demo, fake_obs):
    p = multiprocessing.Pool(16)
    logp, jac = inference.log_p(demo, N, 100000, fake_obs, 1e-7, 1e-8, p)
    eps = .02
    demo2 = Demography(demo.sqrt_a, demo.b + eps, demo.sqrt_s, demo.hs)
    logp2, _ = inference.log_p(demo2, N, 100000, fake_obs, 1e-7, 1e-8, p)
    print(logp2, logp, jac, logp2 - (logp + eps * jac[1, 0]))
