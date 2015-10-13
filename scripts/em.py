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
import psmcpp.scrm, psmcpp.bfgs, psmcpp._pypsmcpp, psmcpp.util, _newick

def kl(sfs1, sfs2):
    s1 = sfs1.flatten()
    s2 = sfs2.flatten()
    nz = s1 != 0.0
    return (s1[nz] * (np.log(s1[nz]) - np.log(s2[nz]))).sum()

block_size = 100
np.set_printoptions(linewidth=120, precision=6, suppress=True)

try:
    progress = bool(int(sys.argv[4]))
except:
    progress = False

try:
    diagnostic = bool(int(sys.argv[5]))
except:
    diagnostic = False

flat = True

psmcpp._pypsmcpp.do_progress(progress)

# 1. Generate some data. 
n = int(sys.argv[1])
N0 = 10000.
rho = 1e-9
theta = 2.5e-8
L = int(float(sys.argv[3]))
ALPHA_PENALTY = 1e5
LAMBDA_PENALTY = 1.0

# PSMC sample demography
a0 = np.array([2.7, .2, 1.5, 2.7])
b0 = a0
s0 = np.array([30000., 70000., 3.5e6 - 1e5, 10000.]) / 25.0 / (2 * N0)

# MSMC sample demography
# a0 = np.array([7.1, 7.1, 0.9, 7.1, 0.9, 7.1, 0.9])
# b0 = np.array([7.1, 0.9, 7.1, 0.9, 7.1, 0.9, 0.9])
# s0 = np.array([1000.0, 4000.0 - 1000., 10500. - 4000., 65000. - 10500., 115000. - 65000., 1e6 - 115000, 1.0]) / 25.0 / (2 * N0)
# # 
# Humanish
# a0 = np.array([8.0, 0.5, 2.0, 1.0])
# b0 = np.array([1.0, 0.5, 2.0, 1.0])
# s0 = np.array([10000., 20000., 50000., 1.0]) / 25. / (2 * N0)

true_parameters = (a0, b0, s0)
print("true parameters")
print(np.array(true_parameters))
demography = psmcpp.scrm.demography_from_params((a0 * 2.0, b0 * 2.0, s0))
print(demography)

# np.save("test/obs_list.npy", obs_list[0])
print("simulating")
data = None
def perform_sim(args):
    n, N0, theta, rho, L, demography, include_trees, seed = args
    global data
    data = psmcpp.scrm.simulate(n, N0, theta, rho, L, demography, include_trees, seed)
    return psmcpp.scrm.hmm_data_format(data, (0, 1))
# obs_list = [perform_sim((n, N0, theta, rho, L, demography, diagnostic, np.random.randint(0, sys.maxint)))]

if diagnostic:
# Extract true hidden states (1-2 coal time) from data
    data = psmcpp.scrm.simulate(n, N0, theta, rho, L, demography, True, np.random.randint(0, sys.maxint))
    obs_list = [psmcpp.scrm.hmm_data_format(data, (0, 1))]
    ct = [(c1, _newick.tmrca(c2, "1", "2")) for c1, c2 in data[3]]
    cts = np.zeros(L)
    fs = frozenset([0,1])
    i = 0
    for span, tmrca in ct:
        cts[i:(i+span)] = tmrca
        i += span
else:
    import multiprocessing
    p = multiprocessing.Pool(16)
    obs_list = list(p.imap_unordered(perform_sim, 
        [(n, N0, theta, rho, L, demography, False, np.random.randint(0, sys.maxint))
            for _ in range(int(sys.argv[2]))]))
    p.terminate()
    p.join()
    del p


# obs_list = []
# for k in range(int(sys.argv[2])):
#     data = psmcpp.scrm.simulate(n, N0, theta, rho, L, demography, False, seed=np.random.randint(0, sys.maxint))
#     obs_list.append(psmcpp.scrm.hmm_data_format(data, (0, 1)))
#     sys.stdout.write(str(k) + " ")
#     sys.stdout.flush()
print("finished simulation")

osfs = []
for ol0 in obs_list:
    obsfs = np.zeros([3, n - 1])
    for r, c1, c2 in ol0[ol0[:, 1:].sum(axis=1) > 0]:
        obsfs[c1, c2] += r
    obsfs /= L
    obsfs[0, 0] = 1. - obsfs.sum()
    osfs.append(obsfs)
obsfs = np.mean(osfs, axis=0)
print(" - Observed sfs:")
print(obsfs)

# 4. Optimize this function
hidden_states = np.array([0., 14.0]) / 25.0 / (2 * N0)
im = psmcpp._pypsmcpp.PyInferenceManager(n - 2, [obs_list[0][:10]], [0.0, 1.0],
        4.0 * N0 * theta, 4.0 * N0 * rho,
        block_size, 5, [0])


# Emission mask
T_MIN = 1000. / 25 / (2 * N0)
T_MAX = 30.0
ni = 40
hs1 = np.zeros(ni)
hs1 = np.concatenate(([0], np.logspace(np.log10(T_MIN), np.log10(T_MAX), ni)))
# hs1 = im.balance_hidden_states((a0,b0,s0),32)
# hs1[:8] = hs2[:8]
# hs2 = im.balance_hidden_states((a0,b0,s0), 64)
# hs1[8:16] = hs2[
print("hidden states", hs1)
em = np.arange(3 *  (n - 1), dtype=int).reshape([3, n - 1])
em[0] = em[2] = 0
em[1] = 1

T_MIN = 10000. / 25 / (2 * N0)
T_MAX = np.cumsum(s0)[-1] * 1.2
ni = 30
s = np.logspace(np.log10(T_MIN), np.log10(T_MAX), ni)
s = np.concatenate(([T_MIN], s[1:] - s[:-1]))
# merge the last three
print(s)
print(np.cumsum(s))

# a=np.array([2.722923,2.722457,2.56506,2.174305,1.613691,1.781124,1.360913,0.851928,0.109846,0.1,0.102629,0.195247
#     ,1.307067,2.128009,2.33168,2.361934,2.421301,2.082566,1.946674,2.08935,2.068704,2.221743,2.100478,1.862589
#     ,1.983183,1.930055,1.930772,1.936481,1.953274,2.755091])
# b=a
# s=np.array([ 0.02    ,  0.004634,  0.005707,  0.00703 ,  0.008658,  0.010664,  0.013135,  0.016178,  0.019926,  0.024543,
#     0.030229,  0.037233,  0.045859,  0.056484,  0.06957 ,  0.085688,  0.105541,  0.129993,  0.160111,  0.197206,
#     0.242896,  0.299171,  0.368484,  0.453857,  0.559008,  0.688522,  0.848042,  1.044521,  1.286521,  1.584588])
# hs1 = np.array([
#     0.      ,   0.002   ,   0.002559,   0.003275,   0.004191,   0.005362,   0.006862,   0.00878 ,   0.011235,
#     0.014377,   0.018397,   0.023541,   0.030123,   0.038546,   0.049324,   0.063116,   0.080764,   0.103347,
#     0.132244,   0.169222,   0.216539,   0.277086,   0.354564,   0.453705,   0.580568,   0.742903,   0.95063 ,
#     1.21644 ,   1.556575,   1.991817,   2.548758,   3.261429,   4.173373,   5.340309,   6.833539,   8.744298,
#     11.189333,  14.318037,  18.321572,  23.444555,  30.      ])

t_start = time.time()
im = psmcpp._pypsmcpp.PyInferenceManager(n - 2, obs_list, hs1,
        4.0 * N0 * theta, 4.0 * N0 * rho,
        block_size, n, [0], em)
im.setParams((a0,b0,s0),False)
im.Estep()
ll_true = np.sum(im.loglik(0.0))

K = len(s)
# x0 = np.array([a, b])
# # a = np.random.uniform(1.0, 9.0, K)
x0 = np.ones([2, K])
a, b = x0
if flat:
    b = a
else:
    # b = np.random.uniform(1.0, 9.0, K)
    b += 0.1

# array = np.array
# d = {'a': array([ 3.966221,  1.916171,  1.856902,  1.07719 ,  0.742571,  0.105161,  0.1     ,  0.1     ,  0.1     ,  0.387956,
#     1.588006,  2.452575,  2.901815,  2.980086,  2.834179,  2.493315,  1.992148,  1.943969,  2.026899,  2.118136,
#     1.798644,  1.762841,  1.820448,  1.824596,  1.81485 ,  1.721289,  1.742825,  1.875612,  1.684476,  1.504643,
#     5.153206]), 'a0': array([ 2.7,  0.2,  1.5,  2.7]), 's': array([ 0.04    ,  0.00767 ,  0.009141,  0.010894,  0.012983,  0.015473,  0.01844 ,  0.021976,  0.02619 ,  0.031213,
#         0.037198,  0.044331,  0.052832,  0.062963,  0.075037,  0.089426,  0.106575,  0.127012,  0.151367,  0.180394,
#         0.214986,  0.256212,  0.305343,  0.363896,  0.433677,  0.516839,  0.615948,  0.734062,  0.874827,  1.042584,
#         1.24251 ]), 'b': array([ 3.966221,  1.916171,  1.856902,  1.07719 ,  0.742571,  0.105161,  0.1     ,  0.1     ,  0.1     ,  0.387956,
#             1.588006,  2.452575,  2.901815,  2.980086,  2.834179,  2.493315,  1.992148,  1.943969,  2.026899,  2.118136,
#             1.798644,  1.762841,  1.820448,  1.824596,  1.81485 ,  1.721289,  1.742825,  1.875612,  1.684476,  1.504643,
#             5.153206]), 'b0': array([ 2.7,  0.2,  1.5,  2.7]), 't_start': 1444673262.255592, 's0': array([ 0.06,  0.14,  6.8 ,  0.02]), 't_now': 1444674247.469531, 'argv': ['scripts/em.py', '10', '20', '5e7']}
# a[:] = d['a']
# b[:] = d['b']

im.setParams((a,b,s),False)
im.Estep()

if diagnostic:
    bk=im.block_keys()[0]
    cs = [sum(d.values()) for _, d in bk]
    xx = np.cumsum(cs)
    g = im.gammas()[0]
    g0 = np.zeros(g.shape)
    I = np.eye(g.shape[0])
    hs2 = hs1.copy()
    hs2[-1] = np.inf
    tcts = np.searchsorted(hs2, cts) - 1
    for i in range(g.shape[1]):
        g0[:,i] = I[tcts[i * block_size]]
    import collections
    d = collections.defaultdict(lambda: collections.Counter())
    dm = collections.defaultdict(lambda: collections.Counter())
    dd = {True: d, False: dm}
    for i in range(g.shape[1]):
        dd[bk[i][0]][tcts[i * block_size]][frozenset(bk[i][1].items())] += 1
    im.setParams((a0,b0,s0),False)
    E = im.emission()
    Em = im.masked_emission()
    F = np.zeros(E.shape)
    Fm = np.zeros(Em.shape)
    Fd = {True: F, False: Fm}
    N = {}
    for b in [True, False]:
        for k in dd[b]:
            for cc in dd[b][k]:
                for cc1, cc2 in cc:
                    Fd[b][k][cc1] += cc2 * dd[b][k][cc]
        N[b] = Fd[b] / Fd[b].sum(axis=1)[:, None]
    Z = np.zeros([len(hs1) - 1, len(hs1) - 1])
    tcts_bs = tcts[::block_size]
    for t1, t2 in zip(tcts_bs[:-1], tcts_bs[1:]):
        Z[t1,t2] += 1
    Zn = Z / Z.sum(axis=1)[:, None]
    T = im.transition()

im.seed = np.random.randint(0, sys.maxint)
llold = -np.inf
Qold = -np.inf

bounds = np.array([[0.1, 20.0]] * 2 * K).reshape([2, K, 2])

def optimize_pg():
    def f(x):
        global s
        aa = x[:K]
        if flat:
            bb = aa
        else:
            bb = x[K:]
        im.setParams((aa, bb, s), False)
        res = im.Q(LAMBDA_PENALTY)
        ret = -np.mean(res, axis=0)
        # penalty
        # esfs = psmcpp._pypsmcpp.sfs(n, (x, x, s), 0.0, np.inf, 4 * N0 * theta, False)
        # diff = esfs[0, 0] - obsfs[0, 0]
        # penalty = ALPHA_PENALTY * diff**2
        # print('penalty', penalty, ret)
        # ret += penalty
        return ret

    def fprime(x):
        global s
        aa = x[:K]
        if flat:
            bb = aa
        else:
            bb = x[K:]
        im.setParams((aa, bb, s), grads)
        res = im.Q(LAMBDA_PENALTY)
        lls = np.array([ll for ll, jac in res])
        jacs = np.array([jac for ll, jac in res])
        #ret = -np.mean(jacs, axis=0)
        ret = [-np.mean(lls, axis=0), -np.mean(jacs, axis=0)]
        # penaly
        # esfs, jac = psmcpp._pypsmcpp.sfs(n, (aa, bb, s), 0.0, np.inf, 4 * N0 * theta, True)
        # diff = esfs[0, 0] - obsfs[0, 0]
        # penalty = ALPHA_PENALTY * diff**2
        # print('penalty', penalty, ret[0])
        # ret[0] += penalty
        # ret[1] += 2 * ALPHA_PENALTY * diff * jac[0, 0, :K]
        return ret

    def project(x):
        return np.minimum(np.maximum(x, bounds[:,0]), bounds[:,1])
    if flat:
        x = a
    else:
        x = np.concatenate([a, b])
    c = tau = 0.5
    f0, grad = fprime(x)
    p = -grad
    m = p.dot(grad)
    alpha = min([(bounds[i][int(p[i] > 0)] - x[i]) / p[i] for i in range(len(x)) if p[i] != 0])
    f1 = f(project(x + alpha * p))
    while f1 > f0 + alpha * c * m:
        f1 = f(project(x + alpha * p))
        print(f0, f1)
        alpha *= tau
    return project(x + alpha * p)

def optimize_fullgrad(iter, coords, x0):
    print("Optimizing all coordinates...")
    def fprime(x):
        x0c = x0.copy()
        for xx, cc in zip(x, coords):
            x0c[cc] = xx
        global s
        aa, bb = x0c
        if flat:
            bb = aa
        print(aa)
        print(bb)
        print(s)
        im.setParams((aa, bb, s), coords)
        print("done")
        res = im.Q(LAMBDA_PENALTY)
        lls = np.array([ll for ll, jac in res])
        jacs = np.array([jac for ll, jac in res])
        ret = [-np.sum(lls, axis=0), -np.sum(jacs, axis=0)]
        if flat:
            print(ret[1])
        else:
            print(ret[1].reshape([2,K]))
        print(ret[0])
        reg = im.regularizer()
        print("regularizer: ", LAMBDA_PENALTY * reg)
        # add penalty
        esfs, jac = psmcpp._pypsmcpp.sfs(n, (aa, bb, s), 0.0, np.inf, 4 * N0 * theta, coords)
        diff = esfs[0, 0] - obsfs[0, 0]
        penalty = ALPHA_PENALTY * diff**2
        print("penalty", penalty)
        ret[0] += penalty
        ret[1] += 2 * ALPHA_PENALTY * diff * jac[0, 0]
        # print(x)
        # print(ret[0])
        # print(ret[1])
        # print("------")
        return ret
    if iter <= 5:
        factr = 1e11
    elif 5 < iter <= 10:
        factr = 1e10
    else:
        factr = 1e9
    # f0, fp = fprime([x0[cc] for cc in coords])
    # print("gradient check")
    # for i, cc in enumerate(coords):
    #     x0c = x0.copy()
    #     x0c[cc] += 1e-8
    #     f1, _ = fprime([x0c[cc] for cc in coords])
    #     print(i, f1, f0, (f1 - f0) / 1e-8, fp[i])
    # print("gradient check", scipy.optimize.check_grad(lambda x: fprime(x)[0], lambda x: fprime(x)[1], x0))
    res = scipy.optimize.fmin_l_bfgs_b(fprime, [x0[cc] for cc in coords], 
            None, bounds=[tuple(bounds[cc]) for cc in coords], disp=False, factr=factr)
    # print(res)
    return res[0]

break_loop = False
import signal, sys
def print_state():
    global a, b, s, a0, b0, s0
    print(repr({'a': a, 'b': b, 's': s, 'a0': a0, 'b0': b0, 's0': s0, 'argv': sys.argv, 't_start': t_start, 't_now': time.time()}))
def signal_handler(signal, frame):
    print("Terminating optimization...")
    print_state()
    sys.exit(1)
signal.signal(signal.SIGINT, signal_handler)

i = 0
ca = [0]
if not flat:
    ca.append(1)
while i < 20:
    if True or i % 10 == 0:
        coords = [(aa, j) for aa in ca for j in range(K)]
    else:
        iM = (i - 1) % ((K // 4) + 1)
        coords = [(aa, j) for aa in ca for j in range(4 * iM, min(K, 4 * (iM + 1)))]
    ret = optimize_fullgrad(i, coords, x0)
    for xx, cc in zip(ret, coords):
        x0[cc] = xx
    print("************** ITERATION %d ***************" % i)
    print(a)
    print(b)
    im.setParams((a, b, s), False)
    im.Estep()
    ll = np.sum(im.loglik(0.0))
    print(" - Tru loglik:" + str(ll_true))
    print(" - New loglik:" + str(ll))
    print(" - Old loglik:" + str(llold))
    if ll < llold:
        print("*** Log-likelihood decreased")
    llold = ll
    esfs = psmcpp._pypsmcpp.sfs(n, (a,b,s), 0.0, np.inf, 4 * N0 * theta, False)
    print("calculated sfs")
    print(esfs)
    print("observed sfs")
    print(obsfs)
    # print("D(Ob. sfs || Est. sfs):", kl(obsfs, esfs))
    # print("D(Ob. sfs || True sfs):", kl(obsfs, esfs0))
    i += 1

print_state()
