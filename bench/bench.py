#/usr/bin/env python2.7
# Complete example showing how to use the package for inference
from __future__ import division
import numpy as np
import scipy.optimize
import pprint
import multiprocessing
import sys
import itertools
from collections import Counter
import sys
import time
import argparse
import psmcpp._pypsmcpp
from psmcpp.lib.util import config2dict
import ConfigParser as configparser
import cPickle as pickle

# setup stuff
t_start = time.time()
np.set_printoptions(linewidth=120, precision=6, suppress=False)

def exp_quantiles(M, h_M):
    hs = -np.log(1. - np.linspace(0, h_M, M, False) / h_M)
    hs = np.append(hs, h_M)
    hs[0] = 0
    return hs

psmcpp._pypsmcpp.do_progress(False)
smcpp_data = psmcpp.lib.util.parse_text_datasets(sys.argv[1:])
n = smcpp_data['n']
thinning = 25 * n
obs_list = [psmcpp.lib.util.normalize_dataset(ob, thinning) for ob in smcpp_data['obs']]

t_1 = 0.01
t_K = 2.0
Ks = "30*1".split("+")
pieces = []
for piece in Ks:
    try:
        num, span = list(map(int, piece.split("*")))
    except ValueError:
        span = int(piece)
        num = 1
    pieces += [span] * num
s = np.logspace(np.log10(t_1), np.log10(t_K), sum(pieces))
s = np.concatenate(([t_1], s[1:] - s[:-1]))
sp = np.zeros(len(pieces))
count = 0
for i, p in enumerate(pieces):
    sp[i] = s[count:(count+p)].sum()
    count += p
s = sp
print("time points", s)
print(np.cumsum(s))

## Compute hidden states
M = 32
h_M = 10.0
hs = exp_quantiles(M, h_M)
if hs[0] != 0:
    raise Exception("First hidden state interval must begin at 0")
hs = np.unique(np.sort(np.concatenate([hs, np.cumsum(s)])))
print("hidden states", hs)

# Load additional params
N0 = 10000
mu = 1e-8
rho = mu / 4.

lambda_penalty = 0.0
im = psmcpp._pypsmcpp.PyInferenceManager(n - 2, obs_list, hs, 4.0 * N0 * mu, 4.0 * N0 * rho)
im.spanCutoff = 64
K = len(s)
x0 = np.ones([2, K])
a, b = x0
exponential_pieces = []
flat_pieces = [i for i in range(K) if i not in exponential_pieces]
b[:] = a + 0.1
b[flat_pieces] = a[flat_pieces]

im.setParams((a,b,s),False)
st = time.time()
im.Estep()
print("estep time: %f" % (time.time() - st))
print(im.Q(0.0))
