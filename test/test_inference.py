import pytest
import multiprocessing
import numpy as np
import logging

logging.getLogger().setLevel("INFO")

import inference

N = 10
L = 1000
hidden_states = np.array([0.0, 1.0, 2.0, np.inf])

@pytest.fixture
def demo():
    return ([1.0, 2.0], [.0001, -.01], [0.0, 0.5])

@pytest.fixture
def fake_obs():
    ary = []
    for ell in range(L):
        ary.append([np.random.randint(1, 1000), 0, 0])
        d = np.random.randint(0, 3)
        ary.append([1, d, np.random.randint(not d, N + 1 - (d == 2))])
    return np.array(ary)

def test_inference_parallel(demo, fake_obs):
    a, b, s = demo
    ll, jac = inference.loglik(a, b, s, 1, 1, N, [fake_obs,] * 4, 1e-8, 1e-8, hidden_states, numthreads=8, jacobian=True)
    print(ll)
    print(jac)
    # Well, that worked

def test_inference(demo, fake_obs):
    logp, jac = inference.logp(demo.sqrt_a, demo.b, demo.sqrt_s, 1000, 500, N, fake_obs, 1e-8, 1e-8, hidden_states)
    print(logp)
    print(jac)
    # Well, that worked

def test_derivatives(demo, fake_obs):
    logp, jac = inference.logp(demo.sqrt_a, demo.b, demo.sqrt_s, 500, 100, N, fake_obs, 1e-8, 1e-8, hidden_states)
    print(jac)
    eps = 0.01
    I = np.eye(demo.K)
    for k in [0, 1, 2]:
        for ell in range(demo.K):
            if k == 2 and ell == 0:
                continue
            args = [demo.sqrt_a, demo.b, demo.sqrt_s, demo.hs]
            delta = args[k][ell] * eps
            args[k][ell] *= (1 + eps)
            demo2 = Demography(*args)
            logp2, _ = inference.logp(demo2, 100, 1000, N, fake_obs, 1e-8, 1e-8, hidden_states)
            print(k, ell, logp2, logp, delta, logp2 - (logp + delta * jac[k, ell]))
