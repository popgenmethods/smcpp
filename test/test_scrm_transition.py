import pytest
import numpy as np

import _pypsmcpp
import lib.scrm as scrm
from fixtures import *

np.set_printoptions(suppress=True)

N0 = 10000
theta = 1.25e-8
rho = theta / 4.
L = 1e8
demography = []

def _test_derivative():
    array = np.array
    # a = array([ 7.1,  7.1,  0.9,  7.1,  0.9,  7.1,  0.9])
    # b = array([ 7.1,  0.9,  7.1,  0.9,  7.1,  0.9,  0.9])
    s = array([ 0.002,  0.006   ,  0.013   ,  0.109   ,  0.1     ,  1.77    ,  0.000002])
    hs = array([ 0., 0.5, 1.0, 2.0, 5.0, 14.0])
    obs = [np.array([[1, 0, 0], [1, 0, 0]], dtype=np.int32)]
    im = _pypsmcpp.PyInferenceManager(0, obs, hs, 4 * N0 * theta, 4 * N0 * rho, 100, 5, [0])
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
    r = 4 * N0 * rho * (L - 1)
    array = np.array
    a = array([ 7.1,  7.1,  0.9,  7.1,  0.9,  7.1,  0.9])
    b = array([ 7.1,  0.9,  7.1,  0.9,  7.1,  0.9,  0.9])
    s = array([ 0.002   ,  0.006   ,  0.013   ,  0.109   ,  0.1     ,  1.77    ,  0.000002])
    # a = np.array([2., 1.])
    # b = np.array([2., 1.])
    # s = np.array([0.5, 0.5])
    hs = array([0., 0.25, 0.5, 1.0, 2.0, 5.0, 14.9])
    ctre = re.compile(r"^\[([^]]+)\]\(\d:(\d+(\.\d+)?),")
    obs = [np.array([[1, 0, 0], [1, 0, 0]], dtype=np.int32)]
    im = _pypsmcpp.PyInferenceManager(0, obs, hs, 4 * N0 * theta, 4 * N0 * rho, 1, 5, [0])
    for hj in (True, False):
        print(hj)
        im.hj = hj
        im.setParams((a, b, s), False)
        np.set_printoptions(suppress=True, linewidth=140)
        trans = im.transition
        print(trans)
    demo = scrm.demography_from_params((2 * a, 2 * b, s))
    print(demo, r)
    out = scrm.scrm(2, 1, "-r", r, L, '-T', *demo, _iter=True)
    spans = []
    cts = []
    for line in out:
        if line[0] == '[':
            m = ctre.match(line)
            spans.append(int(m.group(1)))
            cts.append(float(m.group(2)))
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

