import pytest
import numpy as np

import _pypsmcpp
import scrm
from fixtures import *

np.set_printoptions(suppress=True)

N0 = 10000
theta = 1e-8
rho = 1e-9
L = 1e9
demography = []


def test_transition():
    import re
    r = 4 * N0 * rho * (L - 1)
    a = np.array([0.5, 1.0])
    b = np.array([0.5, 1.0])
    s = np.array([1.0, 1.0])
    t = np.array([0.0, 0.5, 1.0, 2.0, 20.0])
    hs = t
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
