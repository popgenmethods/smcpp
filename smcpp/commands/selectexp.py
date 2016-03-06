'''Perform model selection to choose period(s) for exponential growth.'''


from __future__ import division
import numpy as np
import scipy.optimize
import multiprocessing
import sys
import itertools
from collections import Counter
import sys
import itertools as it
import ad
import json

from .. import _smcpp, util
from ..model import SMCModel

def init_parser(parser):
    parser.description = "Hi"
    parser.add_argument("-M", type=int, default=15, 
            help="maximum index / earliest t[i] for onset of exponential growth")
    parser.add_argument("-K", type=int, default=2, help="maximum number of exponential pieces")
    parser.add_argument("model", type=str, help="Fitted SMC++ model to analyze", widget="FileChooser")

def _iexpstep(a, v, step, ts):
    # Return \int_t[0]^t[-1] ||a * exp(v * t) - step(t)||^2,
    # where step is a step function with breaks given by ti
    from ad.admath import exp
    assert ts[0] == 0
    ret = 0
    for i in range(len(ts) - 1):
        # add in contribution of \int_cs[i]^cs[i+1] |ai * exp(v * (t - cs[k_last])) - model.a[i]|^2 dt
        ret += (-a**2 * (exp(2 * v * ts[i]) - exp(2 * v * ts[i + 1])) + 
                4 * a * (exp(v * ts[i]) - exp(v * ts[i + 1])) * step[i] + 
                2 * v * step[i]**2 * (ts[i + 1] - ts[i])) / (2 * v)
    # def g(x):
    #     i = max(0, np.searchsorted(ts, x) - 1)
    #     return (a * exp(v * x) - step[i])**2
    # ret1 = scipy.integrate.quad(g, ts[0], ts[-1], points=ts[1:-1])
    # print(ret, ret1)
    return ret

def main(args):
    d = json.load(open(args.model, "rt"))
    model_a, model_b, s = d['_model']
    model_K = len(s)
    cs = np.concatenate([[0], np.cumsum(s)])
    # Compute squared error loss of converting all potential
    # segment to exponential
    from ad.admath import log
    res = {}
    for ks in it.combinations(range(args.M), 2):
        k0, k1 = sorted(ks)
        def _f(x):
            ak, bk = x**2
            v = log(bk / ak) / (cs[k1] - cs[k0])
            return _iexpstep(ak, v, model_a[k0:k1], cs[k0:k1 + 1] - cs[k0])
        _g, _h = ad.gh(_f)
        x0 = 1 + np.random.normal(0., .1, size=2)
        res[(k0, k1)] = scipy.optimize.fmin_bfgs(_f, x0, _g, full_output=True, disp=False)
    aic = {}
    for nc in range(1, args.K + 1):
        for comb in it.combinations(range(1, args.M), nc):
            comb = (0,) + tuple(sorted(comb))
            loss = sum(res[(k0, k1)][1] for k0, k1 in zip(comb[:-1], comb[1:]))
            k = 2 * args.K + model_K - comb[-1]
            aic[comb] = 2 * k + np.log(loss)
    res = sorted(aic.items(), key=lambda x: x[1])
    for k in range(1, args.K + 1):
        subres = [r for r in res if len(r[0]) == k + 1]
        print("Best %d-phase model: %s" % (k, str(subres[0][0])))
    print("Overall best model: %s" % str(res[0][0]))
    print(res)
