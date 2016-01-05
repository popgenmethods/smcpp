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

np.set_printoptions(linewidth=120, precision=6, suppress=False)

def exp_quantiles(M, h_M):
    hs = -np.log(1. - np.linspace(0, h_M, M, False) / h_M)
    hs = np.append(hs, h_M)
    hs[0] = 0
    return hs

parser = argparse.ArgumentParser("smc++")
parser.add_argument("--debug", action="store_true", default=False, help="display a lot of debugging info")
parser.add_argument("--comment", default=None, type=str)
parser.add_argument('config', type=argparse.FileType('r'), help="config file")
parser.add_argument('data', nargs="+", help="data file in SMC++ format")
args = parser.parse_args()

psmcpp._pypsmcpp.do_progress(args.debug)
try:
    smcpp_data = pickle.load(open(args.data[0], "rb"))
except:
    smcpp_data = psmcpp.lib.util.parse_text_datasets(args.data)
obs_list = [ob for ob in smcpp_data['obs'] if ob[:,0].sum() > 1000000]
print(len(obs_list))
n = smcpp_data['n']
config = configparser.SafeConfigParser()
config.readfp(args.config)
print(config2dict(config))

## Calculate observed SFS for use later
def _obsfs_helper(ol):
    obsfs = np.zeros([3, n - 1])
    olsub = ol[np.logical_and(ol[:, 1:3].min(axis=1) != -1, ol[:, -1] == n - 2)]
    for r, c1, c2, _ in olsub:
        obsfs[c1, c2] += r
    return obsfs
pool = multiprocessing.Pool(None)
osfs = list(pool.map(_obsfs_helper, obs_list))
pool.close()
pool.terminate()
pool = None
obsfs = np.sum(osfs, axis=0)
obsfs /= obsfs.sum()
print(" - Observed sfs:")
print(obsfs)

# Emission mask
em = np.arange(3 * (n - 1), dtype=int).reshape([3, n - 1])
# em[0, 3:] = 3
# em[1] = 4
# em[2] = 5

# Model parameters
mbs = config.getfloat('advanced', 'minimum block size')

try:
    ts = np.array(eval(config.get('model parameters', 'ts')))
    s = ts[1:] - ts[:-1]
    K = len(s)
except configparser.NoOptionError:
    t_1 = config.getfloat('model parameters', 't_1')
    t_K = config.getfloat('model parameters', 't_K')
    Ks = config.get('model parameters', 'K').split("+")
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
try:
    hs = np.array(eval(config.get('hidden states', 'hidden states')))
except configparser.NoOptionError:
    M = config.getint('hidden states', 'M')
    h_M = config.getfloat('hidden states', 'h_M')
    hs = exp_quantiles(M, h_M)
if hs[0] != 0:
    raise Exception("First hidden state interval must begin at 0")
hs = np.unique(np.sort(np.concatenate([hs, np.cumsum(s)])))
print("hidden states", hs)

# Load additional params
N0 = config.getfloat('parameters', 'N0')
mu = config.getfloat('parameters', 'mu')
rho = config.getfloat('parameters', 'rho')
block_size = config.getint('advanced', 'block size')

t_start = time.time()
try:
    thinning = config.getint('advanced', 'thinning')
except configparser.NoOptionError:
    thinning = n

try:
    lambda_penalty = config.getfloat("advanced", "lambda penalty")
except configparser.NoOptionError:
    lambda_penalty = 0.0

im = psmcpp._pypsmcpp.PyInferenceManager(n - 2, obs_list, hs,
        4.0 * N0 * mu, 4.0 * N0 * rho,
        block_size, thinning, em)

try:
    im.hj = config.getboolean('advanced', 'use hj')
except configparser.NoOptionError:
    im.hj = True
print("using hj: {hj}".format(hj=im.hj))

K = len(s)
x0 = np.ones([2, K])
a, b = x0
try:
    exponential_pieces = eval(config.get('model parameters', 'exponential pieces'))
except configparser.NoOptionError:
    exponential_pieces = []
flat_pieces = [i for i in range(K) if i not in exponential_pieces]
b[:] = a + 0.1
b[flat_pieces] = a[flat_pieces]

im.setParams((a,b,s),False)
im.Estep()

llold = -np.inf
bounds = np.array([[0.1, 20.0]] * K + [[0.15, 19.9]] * K).reshape([2, K, 2])

# Optimization part
precond = 1. / s
precond[-1] = 1. / (15.0 - np.sum(s))

def optimize_fullgrad(iter, coords, x0, factr=1e9):
    print("Optimizing factr {factr}".format(factr=factr))
    def fprime(x):
        x0c = x0.copy()
        # Preconditioner (sort of)
        for xx, cc in zip(x, coords):
            x0c[cc] = xx * precond[cc[1]]
        global s
        aa, bb = x0c
        bb[flat_pieces] = aa[flat_pieces]
        print(aa)
        print(bb)
        print(s)
        im.setParams((aa, bb, s), coords)
        print("done")
        res = im.Q(lambda_penalty)
        lls = np.array([ll for ll, jac in res])
        jacs = np.array([jac for ll, jac in res])
        ret = [-np.mean(lls, axis=0), -np.mean(jacs, axis=0)]
        dary = np.zeros([2, K])
        for i, cc in enumerate(coords):
            ret[1][i] *= precond[cc[1]]
            dary[cc] = ret[1][i]
        print(dary)
        print(ret[0])
        # reg = im.regularizer()
        # print("regularizer: ", LAMBDA_PENALTY * reg)
        # add penalty
        # esfs, jac = psmcpp._pypsmcpp.sfs(n, (aa, bb, s), 0.0, hs1[-1], 4 * N0 * theta, coords)
        # diff = esfs[0, 0] - obsfs[0, 0]
        # penalty = ALPHA_PENALTY * diff**2
        # print("penalty", penalty)
        # ret[0] += penalty
        # ret[1] += 2 * ALPHA_PENALTY * diff * jac[0, 0]
        # print(x)
        # print(ret[0])
        # print(ret[1])
        # print("------")
        return ret
    # print("gradient check")
    # xx0 = np.array([x0[cc] / precond[cc[1]] for cc in coords])
    # f0, fp = fprime(xx0)
    # for i, cc in enumerate(coords):
    #     x0c = xx0.copy()
    #     x0c[i] += 1e-8
    #     f1, _ = fprime(x0c)
    #     print(i, cc, f1, f0, (f1 - f0) / 1e-8, fp[i])
    res = scipy.optimize.fmin_l_bfgs_b(fprime, [x0[cc] / precond[cc[1]] for cc in coords], 
            None, bounds=[tuple(bounds[cc] / precond[cc[1]]) for cc in coords], disp=False, factr=factr)
    # print(res)
    return np.array([x * precond[cc[1]] for x, cc in zip(res[0], coords)])

break_loop = False
import signal, sys
def print_state():
    global a, b, s
    d = {'a': a, 'b': b, 's': s, 'argv': sys.argv, 't_start': t_start, 't_now': time.time(), 'config': config2dict(config), 'comment': args.comment}
    pprint.pprint(repr(d))
    return d

def signal_handler(signal, frame):
    print("State...")
    print_state()

def reverse_progress(signal, frame):
    args.debug = not args.debug
    psmcpp._pypsmcpp.do_progress(args.debug)

signal.signal(signal.SIGUSR1, reverse_progress)

def run_iteration(i, coords, factr):
    global a
    global b
    global x0
    global llold
    global im
    global hs
    # for j in range(K * di // 3, K * (di + 1) // 3)]
    ret = optimize_fullgrad(i, coords, x0, factr)
    for xx, cc in zip(ret, coords):
        x0[cc] = xx
    print(x0)
    print(list(zip(ret, coords)))
    print(flat_pieces)
    b[flat_pieces] = a[flat_pieces]
    print("************** ITERATION %d ***************" % i)
    print(a)
    print(b)
    if i == 5:
        print("rebalancing hidden states")
        h_M = hs[-1]
        hs = im.balance_hidden_states((a, b, s), M)
        hs[-1] = h_M
        hs = np.unique(np.sort(np.concatenate([hs, np.cumsum(s)])))
        im.hidden_states = hs
        print(hs)
    im.setParams((a, b, s), False)
    im.Estep()
    ll = np.sum(im.loglik(0.0))
    print(" - New loglik:" + str(ll))
    print(" - Old loglik:" + str(llold))
    if ll < llold:
        print("*** Log-likelihood decreased")
    if llold == -np.inf:
        ret = 1.
    else:
        ret = (llold - ll) / llold
    llold = ll
    return ret

i = 0
coords = [(aa, j) for j in range(K) for aa in ((0,) if j in flat_pieces else (0, 1))]
while i < 20:
    run_iteration(i, coords, 1e9)
    esfs = psmcpp._pypsmcpp.sfs(n, (a,b,s), 0.0, hs[-1], 4 * N0 * mu, False)
    print("calculated sfs")
    print(esfs)
    print("observed sfs")
    print(obsfs)
    i += 1
d = print_state()
