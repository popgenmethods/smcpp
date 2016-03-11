import numpy as np
np.set_printoptions(linewidth=120)
import sys

import _pypsmcpp
from fixtures import *

NTHREAD = 1

theta = 2.5e-8
N0 = 10000.0

def test_human():
    # a = np.ones(10) * 1.0
    # b = 2 * a
    # s = np.array([0.1] * K)
    n = 30
    hs = np.array([  0.00000000e+00,   1.57510516e-03,   3.17596524e-03,   4.80342731e-03,   6.45838076e-03,   8.14176017e-03,
        9.85454842e-03,   1.15977801e-02,   1.33725451e-02,   1.51799927e-02,   1.70213360e-02,   1.88978566e-02,
        2.08109099e-02,   2.27619309e-02,   2.47524405e-02,   2.67840526e-02,   2.88584815e-02,   3.09775508e-02,
        3.31432027e-02,   3.53575084e-02,   3.76226799e-02,   3.99410827e-02,   4.23152511e-02,   4.47479036e-02,
        4.72274810e-02,   4.97592618e-02,   5.23568105e-02,   5.50236352e-02,   5.77635326e-02,   6.05806203e-02,
        6.34793740e-02,   6.64646703e-02,   6.95418362e-02,   7.27167060e-02,   7.59956883e-02,   8.36312901e-02,
        1.10520947e-01,   1.40114131e-01,   1.73277272e-01,   2.08013012e-01,   2.45337632e-01,   2.84641402e-01,
        3.28285732e-01,   3.78668953e-01,   4.35628033e-01,   4.90162463e-01,   5.44822781e-01,   6.04826342e-01,
        6.73459593e-01,   7.46587090e-01,   8.24394402e-01,   9.07664905e-01,   9.99618999e-01,   1.10424916e+00,
        1.22240461e+00,   1.35543655e+00,   1.51259093e+00,   1.69639213e+00,   1.92186402e+00,   2.21749096e+00,
        2.59796305e+00,   3.11634984e+00,   1.16301593e+01,   3.93560465e+01])
    hs = hs[-4:]
    hs[0] = 0.
    a, b, s = np.array([[  1.00000000e-01,   1.00000000e-01,   1.00000000e-01,   7.54085946e-01,   8.01119298e-01,   8.86484295e-01,
          8.72504047e-01,   8.81577616e-01,   9.20513695e-01,   9.02499043e-01,   9.49087107e-01,   1.01260822e+00,
          1.13992026e+00,   1.18454456e+00,   9.92366952e-01,   1.04391350e+00,   1.13210193e+00,   1.13331895e+00,
          1.12169752e+00,   1.14881289e+00,   1.22861425e+00,   1.26165709e+00,   1.33427004e+00,   1.38762903e+00,
          1.50598817e+00,   1.69890051e+00,   1.70755013e+00,   1.80862965e+00,   1.86570892e+00,   4.00000000e+01],
       [  1.01000000e-01,   1.00000000e-01,   1.00000000e-01,   7.54085946e-01,   8.01119298e-01,   8.86484295e-01,
          8.72504047e-01,   8.81577616e-01,   9.20513695e-01,   9.02499043e-01,   9.49087107e-01,   1.01260822e+00,
          1.13992026e+00,   1.18454456e+00,   9.92366952e-01,   1.04391350e+00,   1.13210193e+00,   1.13331895e+00,
          1.12169752e+00,   1.14881289e+00,   1.22861425e+00,   1.26165709e+00,   1.33427004e+00,   1.38762903e+00,
          1.50598817e+00,   1.69890051e+00,   1.70755013e+00,   1.80862965e+00,   1.86570892e+00,   4.00000000e+01],
       [  4.58019028e-02,   1.42505264e-02,   1.86843487e-02,   2.44976837e-02,   3.21197445e-02,   1.96327987e-02,
          2.24804884e-02,   2.57412286e-02,   2.94749313e-02,   3.37501985e-02,   3.86455827e-02,   4.42510306e-02,
          5.06695350e-02,   5.80190280e-02,   6.64345470e-02,   7.60707166e-02,   8.71045892e-02,   9.97388982e-02,
          1.14205783e-01,   1.30771054e-01,   1.49739078e-01,   1.71458367e-01,   1.96327987e-01,   2.24804884e-01,
          2.57412286e-01,   2.94749313e-01,   3.37501985e-01,   3.86455827e-01,   4.42510306e-01,   5.06695350e-01]])
    theta = 0.000125
    d_to_test = (0, 1)
    coords = [(aa,bb) for aa in d_to_test for bb in range(len(a))]
    im = _pypsmcpp.PyInferenceManager(n - 2, [np.array([[1, 0, 0, n - 2], [1, 1, 1, n - 2]], dtype=np.int32)],
            hs, theta, theta)
    im.setParams((a,b,s), coords)
    aoeu
    for i in range(1, len(hs)):
        t0 = hs[i - 1]
        t1 = hs[i]
        sfs, jac = _pypsmcpp.sfs(n, (a, b, s), t0, t1, theta, coords)
        jac.shape = (3, n - 1, len(d_to_test), K)
        eps = 1e-8
        I = np.eye(K)
        for ind in d_to_test:
            for k in range(K):
                args = [a, b, s]
                args[ind] = args[ind] + eps * I[k]
                # print(args)
                la, lb, ls = args
                sfs2 = _pypsmcpp.sfs(n, (la, lb, ls), t0, t1, theta, jacobian=False)
                for i in (0, 1, 2):
                    for j in range(n - 1):
                        jaca = jac[i, j, ind, k]
                        j1 = sfs2[i, j]
                        j2 = sfs[i, j] + eps * jaca
                        jacb = (sfs2[i,j] - sfs[i,j]) / eps
                        if abs(jaca - jacb) / jacb > 1e-2:
                            print(ind, k, i, j, sfs[i,j], sfs2[i,j], jaca, jacb)
                        # assert abs(j1 - j2) < eps
    assert False

def test_d():
    a = d['a']
    b = d['b']
    s = d['s']
    K = len(a)
    # a = np.ones(10) * 1.0
    # b = 2 * a
    # s = np.array([0.1] * K)
    n = 10
    t0 = 0.0
    t1 = 14.0
    coords = [(aa, bb) for aa in [0, 1] for bb in range(K)]
    sfs, jac = _pypsmcpp.sfs(n, (a, b, s), t0, t1, 4 * N0 * theta, coords)
    jac.shape = (3, n - 1, 2, K)
    eps = 1e-8
    I = np.eye(K)
    for ind in (0, 1):
        for k in range(K):
            args = [a, b, s]
            args[ind] = args[ind] + eps * I[k]
            # print(args)
            la, lb, ls = args
            sfs2 = _pypsmcpp.sfs(n, (la, lb, ls), t0, t1, 4 * N0 * theta, False)
            for i in (0, 1, 2):
                for j in range(n - 1):
                    jaca = jac[i, j, ind, k]
                    j1 = sfs2[i, j]
                    j2 = sfs[i, j] + eps * jaca
                    print(ind, k, i, j, sfs[i,j], sfs2[i,j], jaca, (sfs2[i,j] - sfs[i,j]) / eps)
                    # assert abs(j1 - j2) < eps
    assert False

def test_fail3():
    t0 = 0.0
    t1 = np.inf
    n = 2
    a, b = np.array(
        [[2.785481, 2.357832, 3.189695, 4.396804, 6.340599, 6.74133, 10., 10., 10., 10., 10., 10., 10., 10., 0.1],
         [2.702762, 2.313371, 3.073819, 4.199494, 7.082639, 7.427921, 9.92296, 10., 10., 10., 10., 10., 10., 10., 1.1]])
    s = [0.001, 0.000733, 0.00127, 0.0022, 0.003812, 0.006606, 0.011447, 0.019836, 0.034371, 0.059557, 0.103199, 0.178822,
            0.30986, 0.536921, 0.930368]
    sfs, jac = _pypsmcpp.sfs(n, (a, b, s), t0, t1, 4 * N0 * theta, jacobian=True)


def test_matching_diff_nodiff(demo):
    from timeit import default_timer as timer
    a, b, s = demo
    n = 5
    start = timer()
    seed = np.random.randint(0, 10000000)
    sfs1, rsfs1, jac = _pypsmcpp.sfs((a, b, s), n, num_samples, 0., np.inf, NTHREAD, 1.0, jacobian=True, seed=seed)
    end = timer()
    print("Run-time with jacobian: %f" % (end - start))
    start = timer()
    sfs2, rsfs2 = _pypsmcpp.sfs((a, b, s), S, M, 0., np.inf, NTHREAD, 1.0, jacobian=False, seed=seed)
    end = timer()
    print("Run-time without: %f" % (end - start))
    print(sfs1)
    print(sfs2)
    print(jac)

def test_correct_const(constant_demo_1, constant_demo_1000):
    # Make as close to constant as possible
    obs = [np.array([[1, 0, 0], [1, 0, 0]], dtype=np.int32)]
    for d, mult in ((constant_demo_1, 1.0), (constant_demo_1000, 1000.0)):
        for n in (2, 3, 10, 20):
            sfs, rsfs, jac = _pypsmcpp.sfs((a, b, s), n, S, M, 0., np.inf, NTHREAD, 1.0, jacobian=True, seed=seed)
            im = _pypsmcpp.PyInferenceManager(n - 2, obs, np.array([0.0, np.inf]), 1e-8, 1e-8, 1, NTHREAD, num_samples)
            sfs = im.sfs(d, 0, np.inf)
            rsfs = _pypsmcpp.reduced_sfs(sfs) 
            print(rsfs)
            rsfs /= rsfs[1:].sum()
            print(rsfs)
            for k in range(1, n):
                expected = (1. / k) / sum([1. / j for j in range(1, n)])
                assert (rsfs[k] - expected) / expected < 1e-2

def test_fail():
    t0 = 0.0
    t1 = np.inf
    n = 100
    a, b = [[10., 10., 10., 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 6.044083],
            [10., 10., 10., 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.1]]
    s = [0.058647, 0.093042, 0.147608, 0.234176, 0.371514, 0.589396, 0.93506, 1.483445, 2.353443, 3.733669]
    sfs, jac = _pypsmcpp.sfs(n, (a, b, s), t0, t1, 4 * N0 * theta, jacobian=True)


def test_fail2():
    t0 = 0.0
    t1 = np.inf
    n = 100
    a = [10., 10., 10., 10., 0.1, 0.1, 0.1, 4.141793, 10., 10.]
    b = [10., 10., 10., 4.513333, 0.1, 0.1, 0.1, 2.601576, 4.991452, 1.1]
    s = [0.058647, 0.093042, 0.147608, 0.234176, 0.371514, 0.589396, 0.93506, 1.483445, 2.353443, 3.733669]
    sfs, jac = _pypsmcpp.sfs(n, (a, b, s), t0, t1, 4 * N0 * theta, jacobian=True)

