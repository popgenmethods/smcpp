from __future__ import division
import numpy as np
import multiprocessing as mp
import logging

import util.scrm as scrm
from smcpp.util import sawtooth, human, truncated_sfs
from smcpp.model import PiecewiseModel
from smcpp import _smcpp

np.set_printoptions(suppress=False, linewidth=120)

theta = 1.25e-8

def _scrm_sfs(args):
    np.random.seed(args[-1])
    return scrm.sfs(*(args[:-1]))

def test_human():
    # Diploid effective population size
    N0 = 10000.
    n = 3
    L = 100000
    a = np.array([1.0, 2.0, 1.0])
    s = np.array([2.5, 2.4, 1.0])
    model = PiecewiseModel(a, s)
    sfs = truncated_sfs(model, n, 10.0)
    sfs *= 2 * N0 * theta
    # print(sfs)
    a = np.array([1.0, 2.0, 1.0, 1e-8])
    s = np.array([2.5, 2.4, 10 - 4.9, 1.0])
    demography = scrm.demography_from_params(np.array([a, a, s * 0.5]))
    # print(demography)
    args = (n - 2, L, N0, theta, demography)
    scrm_sfs = np.mean(list(mp.Pool().map(_scrm_sfs, [
                       args + (np.random.randint(0, 10000),) for _ in range(32)])), axis=0)
    # scrm_sfs = scrm.distinguished_sfs(*args)
    print()
    for s in [sfs, scrm_sfs]:
        print(np.array2string(s, formatter={'float_kind': lambda x: "%.3e" % x}))
