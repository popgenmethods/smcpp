import numpy as np
np.set_printoptions(linewidth=120)
import sys

import _pypsmcpp
from fixtures import *

NTHREAD = 1

theta = 2.5e-8
N0 = 10000.0

array = np.array
d = {'a': array([ 3.966221,  1.916171,  1.856902,  1.07719 ,  0.742571,  0.105161,  0.1     ,  0.1     ,  0.1     ,  0.387956,
    1.588006,  2.452575,  2.901815,  2.980086,  2.834179,  2.493315,  1.992148,  1.943969,  2.026899,  2.118136,
    1.798644,  1.762841,  1.820448,  1.824596,  1.81485 ,  1.721289,  1.742825,  1.875612,  1.684476,  1.504643,
    5.153206]), 'a0': array([ 2.7,  0.2,  1.5,  2.7]), 's': array([ 0.04    ,  0.00767 ,  0.009141,  0.010894,  0.012983,  0.015473,  0.01844 ,  0.021976,  0.02619 ,  0.031213,
        0.037198,  0.044331,  0.052832,  0.062963,  0.075037,  0.089426,  0.106575,  0.127012,  0.151367,  0.180394,
        0.214986,  0.256212,  0.305343,  0.363896,  0.433677,  0.516839,  0.615948,  0.734062,  0.874827,  1.042584,
        1.24251 ]), 'b': array([ 3.966221,  1.916171,  1.856902,  1.07719 ,  0.742571,  0.105161,  0.1     ,  0.1     ,  0.1     ,  0.387956,
            1.588006,  2.452575,  2.901815,  2.980086,  2.834179,  2.493315,  1.992148,  1.943969,  2.026899,  2.118136,
            1.798644,  1.762841,  1.820448,  1.824596,  1.81485 ,  1.721289,  1.742825,  1.875612,  1.684476,  1.504643,
            5.153206]), 'b0': array([ 2.7,  0.2,  1.5,  2.7]), 't_start': 1444673262.255592, 's0': array([ 0.06,  0.14,  6.8 ,  0.02]), 't_now': 1444674247.469531, 'argv': ['scripts/em.py', '10', '20', '5e7']}

def test_d():
    a = d['a']
    b = d['b']
    s = np.cumsum(d['s'])
    K = len(a)
    # a = np.ones(10) * 1.0
    # b = 2 * a
    # s = np.array([0.1] * K)
    n = 10
    t0 = s[0]
    t1 = s[1]
    sfs, jac = _pypsmcpp.sfs(n, (a, b, s), t0, t1, 4 * N0 * theta, jacobian=True)
    jac.shape = (3, n - 1, 3, K)
    eps = 1e-8
    I = np.eye(K)
    for ind in (0, 1, 2):
        for k in range(K):
            args = [a, b, s]
            args[ind] = args[ind] + eps * I[k]
            # print(args)
            la, lb, ls = args
            sfs2 = _pypsmcpp.sfs(n, (la, lb, ls), t0, t1, 4 * N0 * theta, jacobian=False)
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

