import pytest
import numpy as np

import fixtures
import smcpp._smcpp, smcpp.model

@pytest.fixture
def fake_obs():
    L = 10000
    n = 25
    ary = []
    for ell in range(L):
        ary.append([np.random.randint(1, 1000), 0, 0])
        d = np.random.randint(0, 3)
        ary.append([1, d, np.random.randint(not d, n + 1 - (d == 2))])
    return np.array(ary, dtype=np.int32)

@pytest.fixture
def constant_demo_1():
    a = np.array([1.0])
    b = np.array([1.0])
    s = np.array([1.0])
    return (a, b, s)


def test_d(fake_obs, constant_demo_1):
    N0 = 10000.
    theta = 1.25e-8
    rho = theta / 4.
    n = 26
    obs_list = [np.array([[1, 0, 10, n - 2], [10, 0, 0, n - 2], [1, 2, 2, n - 5]], dtype=np.int32)]
    hidden_states = np.array([  0.        ,   0.0557381 ,   0.10195686,   0.12346455,
         0.13234427,   0.14137324,   0.28881064,   0.45318096,
         0.54786669,   0.62940343,   0.70119707,   0.77310611,
         0.82501483,   0.87793155,   0.93189619,   1.06666826,
         1.24549378,   1.4292705 ,   1.60683828,   1.78367717,
         1.79302099,   1.80257963,   1.8123632 ,   1.82238254,
         1.83264929,   1.84317599,   1.85397615,   1.86506434,
         1.87645635,   1.88816931,   1.90022182,   1.91263416,
         1.92542847,   1.93862903,   1.95226246,   1.96635813,
         1.98094849,   1.99606952,   2.01176128,   2.02806857,
         2.04504164,   2.06273718,   2.08121945,   2.10056165,
         2.12084775,   2.1421746 ,   2.16465481,   2.18842027,
         2.21264423,   2.23754309,   2.26416043,   2.29275117,
         2.32363152,   2.35720038,   2.39397089,   2.43461882,
         2.48005933,   2.53157553,   2.59104662,   2.661386  ,
         2.74747443,   2.85846172,   3.01488953,   3.28230463,  14.9       ])
    hidden_states = hidden_states[::5]
    M = hidden_states.shape[0] - 1
    em = np.arange(3 *  (n - 1), dtype=int).reshape([3, n - 1])
    em[0] = em[2] = 0
    em[1] = 1
    print(obs_list)
    im = smcpp._smcpp.PyInferenceManager(n - 2, obs_list, hidden_states)
    im.theta = 4.0 * N0 * theta
    im.rho = 4.0 * N0 * rho
    K = 10
    a = np.ones(K)
    b = a
    s = np.diff(np.logspace(np.log10(.01), np.log10(3.), K + 1))
    print('hs', hidden_states)
    print('cumsum(s)', np.cumsum(s))
    K = len(a)
    eps = 1e-8
    model = smcpp.model.SMCModel(s, [])
    model.x[:] = (a, b, s)
    im.model = model
    print(im.model.x)
    im.derivatives = [(0, k) for k in range(K)]
    im.theta = .00025
    im.rho = .00025
    im.E_step()
    im.Q()
    trans1, jac = im.transition
    jac.shape = (M, M, K)
    I = np.eye(K)
    M = trans1.shape[0]
    im.derivatives = []
    for k in range(K):
        aa = [_ for _ in a]
        aa += eps * I[k]
        model.x[0] = aa
        print(model.x)
        im.model = model
        im.Q()
        trans2 = im.transition
        for i in range(M):
            for j in range(M):
                jaca = jac[i, j, k]
                j1 = trans2[i, j]
                j2 = trans1[i, j] + eps * jaca
                print(k, i, j, jaca, (trans2[i,j] - trans1[i,j]) / eps)
    assert False

# def test_equal_jac_nojac(constant_demo_1, hs):
#     from timeit import default_timer as timer
#     start = timer()
#     trans1, jac = _pypsmcpp.transition(constant_demo_1, hs, rho, True)
#     end = timer()
#     print("Run-time with jacobian: %f" % (end - start))
#     start = timer()
#     trans2 = _pypsmcpp.transition(constant_demo_1, hs, rho, False)
#     end = timer()
#     print("Run-time without: %f" % (end - start))
#     assert np.allclose(trans1 - trans2, 0)
# 
# def test_sum_to_one(constant_demo_1, hs):
#     trans1 = _pypsmcpp.transition(constant_demo_1, hs, rho, False)
#     assert np.allclose(np.sum(trans1, axis=1), 1.0)
