import pytest
import numpy as np

import _pypsmcpp
import lib.scrm as scrm
from lib.util import human
from fixtures import *

np.set_printoptions(suppress=True)

N0 = 10000
theta = 1.25e-8
rho = theta / 4.
L = 1e9
demography = []

def _test_derivative():
    array = np.array
    # a = array([ 7.1,  7.1,  0.9,  7.1,  0.9,  7.1,  0.9])
    # b = array([ 7.1,  0.9,  7.1,  0.9,  7.1,  0.9,  0.9])
    s = array([ 0.002,  0.006   ,  0.013   ,  0.109   ,  0.1     ,  1.77    ,  0.000002])
    hs = array([ 0., 0.5, 1.0, 2.0, 5.0, 14.0])
    obs = [np.array([[1, 0, 0], [1, 0, 0]], dtype=np.int32)]
    im = _pypsmcpp.PyInferenceManager(0, obs, hs, 2 * N0 * theta, 2 * N0 * rho, 100, 5, [0])
    im.setParams((a, b, s), True)
    K = len(a)
    np.set_printoptions(suppress=True)
    im.setParams((a, b, s), True)
    T, Tjac = im.transition
    M = T.shape[0]
    Tjac = Tjac.reshape([M, M, 3, K])
    I = np.eye(K)
    eps = 1e-8
    for ind in (0, 1, 2):
        for k in range(K):
            args = [a, b, s]
            args[ind] = args[ind] + eps * I[k]
            print(args)
            la, lb, ls = args
            im.setParams([la, lb, ls], False)
            T1 = im.transition
            print(T1)
            for i in range(M):
                for j in range(M):
                    jac = Tjac[i,j,ind,k]
                    j1 = T[i,j]
                    j2 = T1[i,j]
                    print(ind, k, i, j, (j2-j1) / eps, jac)

def test_transition1():
    import re
    a = human['a']
    b = human['b']
    s = human['s_gen'] / (2 * N0)
    # a = np.array([2., 1.])
    # b = np.array([2., 1.])
    # s = np.array([0.5, 0.5])
    hs = np.array([0.0, 0.02265082630305623, 0.029053078518935623, 0.0372649262386312, 0.04393139764032177, 0.04779785132462101, 0.06130790590113054, 0.06943373637853123, 0.07672122046331377, 0.07863657511411541, 0.08905917019885312, 0.10086318974342721, 0.11062277213899561, 0.11423172956252865, 0.12937215323092655, 0.14571409195026652, 0.14651930856430462, 0.1659391704166982, 0.18208173612093936, 0.18793296629909392, 0.21284184881294496, 0.21982206410398591, 0.24105218737416856, 0.2730015613092028, 0.29808555451453456, 0.3091855472838956, 0.3501654063471451, 0.3797295435550447, 0.39657678982541433, 0.44913959910795487, 0.4648487723926368, 0.50866915226093, 0.5537522975343044, 0.5760888306792596, 0.64679232880409, 0.6524444019423341, 0.7389202410398605, 0.7443726571429539, 0.8368577015803166, 0.8469592459180549, 0.9477759219435313, 0.955093688458607, 1.0694105161385048, 1.0733953891080954, 1.1906597597713753, 1.215664625659446, 1.3197368020465206, 1.3767904139291, 1.457722545020432, 1.5592720260810475, 1.6059384893279027, 1.7659399910988187, 1.7660239046749353, 1.940046658654195, 2.0, 2.1306670182628444, 2.3413880495784976, 2.5769541208912643, 2.84401690614031, 3.1523182657948343, 3.51696137938298, 3.963248482011155, 4.538612626914716, 5.349542843131045, 6.735837204250935, 49.999])
    hs = hs[:8]
    hs[-1] = 49.9
    print(hs)
    ctre = re.compile(r"^\[([^]]+)\]\(\d:(\d+(\.\d+)?),")
    obs = [np.array([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=np.int32)]
    im = _pypsmcpp.PyInferenceManager(0, obs, hs, 2 * N0 * theta, 2 * N0 * rho)
    im.setParams((a, b, s), False)
    np.set_printoptions(suppress=True, linewidth=140)
    trans = im.transition
    print(trans)
    np.savetxt('trans1.txt', trans)
    demo = scrm.demography_from_params((a, b, s * 0.5))
    r = 4 * N0 * rho * (L - 1)
    print(demo, r)
    out = scrm.scrm(2, 1, "-l", 0, "-r", r, L, '-T', *demo, _iter=True)
    spans = []
    cts = []
    for line in out:
        if line[0] == '[':
            m = ctre.match(line)
            spans.append(int(m.group(1)))
            cts.append(2. * float(m.group(2)))
    cts = np.array(cts)
    spans = np.array(spans)
    ctis = np.minimum(np.searchsorted(hs, cts) - 1, len(hs) - 2)
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
    print(C)
    np.savetxt('trans2.txt', C)
    np.savetxt('trans3.txt', C / trans)
    aoeu

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

