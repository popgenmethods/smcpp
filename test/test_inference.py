import pytest
import multiprocessing
import numpy as np
import logging
import _pypsmcpp
import sys

logging.getLogger().setLevel("INFO")

from fixtures import *

n = 50
L = 1000

@pytest.fixture
def fake_obs():
    ary = []
    for ell in range(L):
        ary.append([np.random.randint(1, 1000), 0, 0])
        d = np.random.randint(0, 3)
        ary.append([1, d, np.random.randint(not d, n + 1 - (d == 2))])
    return np.array(ary, dtype=np.int32)

hidden_states = np.array([  0.      ,   0.110721,   0.474051,   1.564003,   2.951359,   4.592253,   6.600545,   9.189684,  12.83887 ,
            19.077194,        np.inf])
num_threads = 1
num_samples = 10
block_size = 50

def test_derivatives(demo, fake_obs):
    N0 = 10000.
    rho = 1e-9
    theta = 2.5e-8
    fake_obs = np.load("test/obs_list.npy")
    obs_list = [fake_obs]
    em = np.arange(3 *  (n - 1), dtype=int).reshape([3, n - 1])
    em[0] = em[2] = 0
    em[1] = 1
    im = _pypsmcpp.PyInferenceManager(n - 2, obs_list, hidden_states, 
            4.0 * N0 * theta, 4.0 * N0 * rho, block_size, num_threads, num_samples, 50, [0], em)
    def f(x):
        # print("f", x, recompute)
        im.setParams(x, False)
        return -im.Q(0.)[0][0]
    a = np.array([   2.640374,    5.552209,   44.405643,   37.321315,    5.466158,    2.024763,  100.      ,  100.      ,
                100.      ,    6.634416])
    s = np.array([0.1] * 10)
    demo = (a, a, s)
    im.setParams(demo, True)
    im.Estep()
    ret = im.Q(0.0)[0]
    q = -ret[0]
    K = demo[0].shape[0]
    jac = -ret[1].reshape(3, K)
    print(jac)
    eps = 1e-8
    K = len(demo[0])
    I = np.eye(K)
    for k in [0, 1, 2]:
        for ell in range(K):
            args = [demo[0].copy(), demo[1].copy(), demo[2].copy()]
            args[k][ell] += eps
            print(args)
            q2 = f(args)
            print(k, ell, q2, q, jac[k, ell], (q2 - q) / eps)
    assert False

def test_diff_nodiff(demo, fake_obs):
    N0 = 10000.
    rho = theta = 1e-8
    obs_list = [fake_obs]
    im = _pypsmcpp.PyInferenceManager(n, obs_list, hidden_states, 
            4.0 * N0 * theta / 2.0, 4.0 * N0 * rho, block_size, num_threads, num_samples)
    for ad in [True, False]:
        im.set_seed(1)
        im.setParams(demo, ad)
        im.Estep()
    ret = im.Q(0.0, False)[0]
    ret2, _ = im.Q(0.0, True)[0]
    assert abs(ret - ret2) < 1e-10

def test_inference_parallel(constant_demo_1, fake_obs):
    a, b, s = constant_demo_1
    ll, jac = inference.loglik(a, b, s, 1, 1, N, [fake_obs,] * 4, 1e-8, 1e-8, hidden_states, numthreads=8, jacobian=True)
    print(ll)
    print(jac)
    # Well, that worked
