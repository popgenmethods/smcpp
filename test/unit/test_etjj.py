from __future__ import division
import warnings
# warnings.filterwarnings('error')
import numpy as np 
from scipy.integrate import quad, dblquad, simps 
from scipy.special import comb as binom
from scipy.interpolate import PPoly
from util import memoize
import pytest

import moran_eigensystem
import etjj

def piecewise_constant(Ninv, ts):
    assert ts[0] == 0.
    assert ts[-1] == np.inf
    K = len(Ninv)
    eta = PPoly([Ninv], ts)
    R = eta.antiderivative()
    r = R(ts[:-1])
    return eta, R, r, K

N = 10
Ninv = np.array([001., 0.01, .005, 0.0002, .000018, .002])
ts = np.array([0., .0001, 0.1, 0.2, .5, 1.0, np.inf])
eta, R, r, K = piecewise_constant(Ninv, ts)
sps = [0.01, 0.05, 0.08, 0.09999, 0.1, 0.19, 0.2, 0.21, 0.25, 0.3, 0.8, 1.0, 2.0, 5.0]
# sps = sps[-4:]

def test_etjj_above_bug1():
    Ninv = np.array([1.0])
    tss = np.array([0., np.inf])
    sps = np.array([])
    etjj_above = etjj.etjj_above(5, Ninv, tss, sps)
    assert np.all(etjj_above > 0)

def test_etjj_below_E2():
    etjj_below_E2 = etjj.etjj_below_E2(N, Ninv, ts, sps)
    cjs = binom(np.arange(2, N + 3), 2)
    for i, sp in enumerate(sps + [np.inf]):
        for j, cj in enumerate(cjs):
            def _f(tau):
                return tau * eta(tau) * np.exp(-cj * R(tau))
            nint = quad(_f, 0., sp)
            print(i, j, sp, cj, etjj_below_E2[i, j], nint)
            assert abs(etjj_below_E2[i, j] - nint[0]) < 1e-5

def test_etjj_below_E1():
    etjj_below_E1 = etjj.etjj_below_E1(N, Ninv, ts, sps)
    cjs = binom(np.arange(2, N + 3), 2)
    for i, sp in enumerate(sps + [np.inf]):
        for j, cj in enumerate(cjs):
            dj = cj - 1
            def _f(s, tau):
                return s * dj * eta(s) * np.exp(-dj * R(s)) * eta(tau) * np.exp(-R(tau))
            nint = dblquad(_f, 0., sp, lambda tau: 0, lambda tau: tau)
            print(i, j, sp, cj, etjj_below_E1[i, j], nint)
            assert abs(etjj_below_E1[i, j] - nint[0]) < 1e-5

def test_etjj_above():
    j = np.arange(2, N + 2)
    cjs = (j * (j - 1) / 2)[:, None]
    lams = moran_eigensystem.eigenvalues(N)
    eta, R, r, K = piecewise_constant(Ninv, ts)
    etjj_above = etjj.etjj_above(N, Ninv, ts, sps)
    for i, sp in enumerate(sps + [50.0]):
        for j, cj in enumerate(cjs):
            for k, lam in enumerate(lams):
                print(sp, cj, lam)
                def _f(s, tau):
                    return (s - tau) * cj * eta(s) * np.exp(-cj * (R(s) - R(tau))) * eta(tau) * np.exp((lam - 1) * R(tau))
                nint = dblquad(_f, 0., sp, lambda x: x, lambda x: np.inf)
                print(i,j,k, etjj_above[i,j,k], nint)
                assert abs(etjj_above[i, j, k] - nint[0]) < 1e-5


def test_trivial_case():
    Ninv = np.array([1.])
    ts = np.array([0., np.inf])
    cj = 1
    etjj.etjj_below(Ninv, [cj], ts, [1.])
    etjj.etjj_above(Ninv, [cj], [-1], ts, [1.])


def test_lower_limit():
    ts = np.array([0., np.inf])
    Ninv = np.array([1.])
    for j in range(3, 20):
        cj = j * (j - 1) / 2
        a1 = etjj.etjj_below(Ninv, cj, ts, [])
        assert abs(a1 - 1 / cj) < 1e-8
