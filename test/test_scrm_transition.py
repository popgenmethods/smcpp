import pytest
import numpy as np

import _pypsmcpp
import scrm
from fixtures import *

np.set_printoptions(suppress=True)

N0 = 10000
theta = 1e-8
rho = 1e-9
L = 1e7
demography = []

def test_derivative():
    a=np.array([2.722923,2.722457,2.56506,2.174305,1.613691,1.781124,1.360913,0.851928,0.109846,0.1,0.102629,0.195247
        ,1.307067,2.128009,2.33168,2.361934,2.421301,2.082566,1.946674,2.08935,2.068704,2.221743,2.100478,1.862589
        ,1.983183,1.930055,1.930772,1.936481,1.953274,2.755091])
    b=a
    s=np.array([ 0.02    ,  0.004634,  0.005707,  0.00703 ,  0.008658,  0.010664,  0.013135,  0.016178,  0.019926,  0.024543,
        0.030229,  0.037233,  0.045859,  0.056484,  0.06957 ,  0.085688,  0.105541,  0.129993,  0.160111,  0.197206,
        0.242896,  0.299171,  0.368484,  0.453857,  0.559008,  0.688522,  0.848042,  1.044521,  1.286521,  1.584588])
    hs = np.array([
        0.      ,   0.002   ,   0.002559,   0.003275,   0.004191,   0.005362,   0.006862,   0.00878 ,   0.011235,
        0.014377,   0.018397,   0.023541,   0.030123,   0.038546,   0.049324,   0.063116,   0.080764,   0.103347,
        0.132244,   0.169222,   0.216539,   0.277086,   0.354564,   0.453705,   0.580568,   0.742903,   0.95063 ,
        1.21644 ,   1.556575,   1.991817,   2.548758,   3.261429,   4.173373,   5.340309,   6.833539,   8.744298,
        11.189333,  14.318037,  18.321572,  23.444555,  30.      ])
    obs = [np.array([[1, 0, 0], [1, 0, 0]], dtype=np.int32)]
    im = _pypsmcpp.PyInferenceManager(0, obs, hs, 4 * N0 * theta, 4 * N0 * rho, 100, 5, [0])
    im.setParams((a, b, s), True)
    K = len(a)
    np.set_printoptions(suppress=True)
    T, Tjac = im.transition()
    im.setParams((a / 2., b, s), True)
    T, Tjac = im.transition()
    im.setParams((a, b, s), True)
    T, Tjac = im.transition()
    print(T[:5,:5])
    print(T[14])
    print(T[15])
    print(T[16])
    aoeu

    aoeu
    M = T.shape[0]
    Tjac = Tjac.reshape([M, M, 3, K])
    I = np.eye(K)
    eps=1e-8
    for ind in (0, 1, 2):
        for k in range(K):
            args = [a, b, s]
            args[ind] = args[ind] + eps * I[k]
            print(args)
            la, lb, ls = args
            im.setParams([la, lb, ls], False)
            T1 = im.transition()
            print(T1)
            for i in range(M):
                for j in range(M):
                    jac = Tjac[i,j,ind,k]
                    j1 = T[i,j]
                    j2 = T1[i,j]
                    print(ind, k, i, j, (j2-j1) / eps, jac)

def _test_transition1():
    import re
    r = 4 * N0 * rho * (L - 1)
    a,b,s = np.array([[ 7.1     ,  7.1     ,  0.9     ,  7.1     ,  0.9     ,  7.1     ,  0.9     ],
        [ 7.1     ,  0.9     ,  7.1     ,  0.9     ,  7.1     ,  0.9     ,  0.9     ],
        [ 0.002   ,  0.006   ,  0.013   ,  0.109   ,  0.1     ,  1.77    ,  0.000002]])
    # a = np.array([8.0, 0.5, 2.0, 1.0])
    # b = np.array([1.0, 0.5, 2.0, 1.0])
    # s = np.array([10000., 20000., 50000., 1.0]) / 25. / (2 * N0)
    #hs = np.array([  0.      ,   0.094589,   0.133865,   0.189422,   0.398023,   0.587645,   0.748156,   0.888184,   1.013083,
    #    1.126418,   1.230689,   1.327726,   1.418919,   1.505357,   1.587916,   1.667328,   1.744216,   1.819131,
    #    1.892576,   1.96503 ,   2.037772,   2.116083,   2.201862,   2.296686,   2.402691,   2.522869,   2.661605,
    #    2.825694,   3.026523,   3.285437,   3.650356,   4.274188,  20.      ])
    hs = np.array([0., 1., 2., 3., 8., 10., 20.])
    ctre = re.compile(r"^\[([^]]+)\]\(\d:(\d+(\.\d+)?),")
    demo = scrm.demography_from_params((2 * a, 2 * b, s))
    print(demo, r)
    out = scrm.scrm(2, 1, "-r", r, L, '-T', '-l', 0, *demo, _iter=True)
    spans = []
    cts = []
    for line in out:
        if line[0] == '[':
            m = ctre.match(line)
            spans.append(int(m.group(1)))
            cts.append(float(m.group(2)))
    cts = np.array(cts)
    spans = np.array(spans)
    ctis = np.searchsorted(hs, cts) - 1
    M = np.zeros([len(hs) - 1, len(hs) - 1])
    C = np.zeros(M.shape)
    P = np.zeros([len(hs) - 1, 2])
    for i in range(len(ctis) - 1):
        M[ctis[i], ctis[i + 1]] += 1
        C[ctis[i], ctis[i]] += spans[i] - 1
        C[ctis[i], ctis[i + 1]] += 1 
        P[ctis[i], 0] += spans[i] - 1
        P[ctis[i], 1] += 1
    M /= M.sum(axis=1)[:,None]
    C /= C.sum(axis=1)[:, None]
    # print(P)
    obs = [np.array([[1, 0, 0], [1, 0, 0]], dtype=np.int32)]
    im = _pypsmcpp.PyInferenceManager(0, obs, hs, 4 * N0 * theta, 4 * N0 * rho, 1, 5, [0])
    im.setParams((a, b, s), False)
    np.set_printoptions(suppress=True, linewidth=140)
    print(M)
    trans = im.transition()
    print(trans[:10,:10])
    print(C[:10,:10])
    print(trans[-10:,-10:])
    print(C[-10:,-10:])
    print(trans)
    print(C)

def no_test_transition():
    import re
    r = 4 * N0 * rho * (L - 1)
    a = np.array([8.0, 0.5, 2.0, 1.0])
    b = np.array([1.0, 0.5, 2.0, 1.0])
    s = np.array([10000., 20000., 50000., 1.0]) / 25. / (2 * N0)
    T_MAX = 30.0
    ni = 5
    hs = 0.1 * np.expm1(np.arange(ni + 1) * np.log1p(10 * T_MAX) / ni)
    ctre = re.compile(r"^\[([^]]+)\]\(\d:(\d+(\.\d+)?),")
    demo = scrm.demography_from_params((2 * a, 2 * b, s))
    print(demo, r)
    out = scrm.scrm(2, 1, "-r", r, L, '-T', '-l', 0, *demo, _iter=True)
    spans = []
    cts = []
    for line in out:
        if line[0] == '[':
            m = ctre.match(line)
            spans.append(int(m.group(1)))
            cts.append(float(m.group(2)))
    cts = np.array(cts)
    spans = np.array(spans)
    ctis = np.searchsorted(hs, cts) - 1
    M = np.zeros([len(hs) - 1, len(hs) - 1])
    C = np.zeros(M.shape)
    P = np.zeros([len(hs) - 1, 2])
    for i in range(len(ctis) - 1):
        M[ctis[i], ctis[i + 1]] += 1
        C[ctis[i], ctis[i]] += spans[i] - 1
        C[ctis[i], ctis[i + 1]] += 1 
        P[ctis[i], 0] += spans[i] - 1
        P[ctis[i], 1] += 1
    print(M)
    print(C / C.sum(axis=1)[:, None])
    print(P)
    obs = [np.array([[1, 0, 0], [1, 0, 0]], dtype=np.int32)]
    im = _pypsmcpp.PyInferenceManager(0, obs, hs, 4 * N0 * theta, 4 * N0 * rho, 1, 5, [0])
    im.setParams((a, b, s), False)
    trans = im.transition()
    print(trans)

