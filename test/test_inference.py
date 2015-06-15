import pytest
import multiprocessing
import numpy as np
import logging
import _pypsmcpp
import sys

logging.getLogger().setLevel("INFO")

import inference

from fixtures import *

n = 10
L = 10

@pytest.fixture
def fake_obs():
    ary = []
    for ell in range(L):
        ary.append([np.random.randint(1, 1000), 0, 0])
        d = np.random.randint(0, 3)
        ary.append([1, d, np.random.randint(not d, n + 1 - (d == 2))])
    return np.array(ary)


hidden_states = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0, np.inf])
NTHREADS = 1

def test_derivatives(demo, fake_obs):
    S = 1000
    M = 10
    N0 = 10000.
    rho = theta = 1e-8
    obs_list = [fake_obs]
    block_size=20
    def Q(x, jacobian, recompute):
        a, b, s = x
        if (recompute):
            update_seed()
        _pypsmcpp.set_csfs_seed(Q._seed)
        ret = _pypsmcpp.Q(
                (a, b, s),
                n,
                S, M, # Parameters which govern accuracy with which
                # the SFS is numerically computed.
                obs_list, # List of the observations datasets we prepared above
                hidden_states,
                4. * N0 * rho * block_size, 4. * N0 * theta / 2., # Same parameters as above
                0.0,
                block_size,
                NTHREADS, # Using multiple threads speeds everything up.
                mw,
                jacobian=jacobian,
                recompute=recompute
                ) 
        return ret
    def update_seed():
        Q._seed = 1 # np.random.randint(0, sys.maxint)
    def f(x, recompute=False):
        # print("f", x, recompute)
        ret = -Q(x, False, recompute)
        # print("f(%s) = %f" % (str(x), ret))
        return ret
    def fprime(x, recompute=False):
        ret = -Q(x, True, recompute)[1]
        # print("f'(%s) = %s" % (str(x), str(ret)))
        return ret
    jac = fprime(demo, True)
    q = f(demo, False)
    print(jac)
    eps = 2e-6
    K = len(demo[0])
    I = np.eye(K)
    for k in [0, 1, 2]:
        for ell in range(K):
            args = [demo[0].copy(), demo[1].copy(), demo[2].copy()]
            print(args)
            args[k][ell] += eps
            q2 = f(args, True)
            print(k, ell, q2, q, jac[k, ell], (q2 - q) / eps)
    assert False

def test_loglik_diff_nodiff(demo, fake_obs):
    a, b, s = demo
    S = 100
    M = 10
    N0 = 1000.
    rho = theta = 1e-8
    obs_list = [fake_obs]
    def f(a, b, s, jacobian):
        return inference.loglik(
                [a, b, s],
                N,
                S, M,
                obs_list, # List of the observations datasets we prepared above
                hidden_states,
                N0 * rho, # Same parameters as above
                N0 * theta, 
                reg=0.,
                numthreads=NTHREADS, seed=1, viterbi=False, jacobian=jacobian)
    logp, jac = f(a, b, s, True)
    logp2 = f(a, b, s, False)
    print(logp, logp2)
    print(jac)
    aoeu

def test_inference_parallel(constant_demo_1, fake_obs):
    a, b, s = constant_demo_1
    ll, jac = inference.loglik(a, b, s, 1, 1, N, [fake_obs,] * 4, 1e-8, 1e-8, hidden_states, numthreads=8, jacobian=True)
    print(ll)
    print(jac)
    # Well, that worked
