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

import psmcpp.scrm, psmcpp.bfgs, psmcpp._pypsmcpp, psmcpp.util, _newick

def kl(sfs1, sfs2):
    s1 = sfs1.flatten()
    s2 = sfs2.flatten()
    nz = s1 != 0.0
    return (s1[nz] * (np.log(s1[nz]) - np.log(s2[nz]))).sum()

block_size = 50
np.set_printoptions(linewidth=120, precision=6, suppress=True)

psmcpp._pypsmcpp.do_progress(False)
diagnostic = False

# 1. Generate some data. 
n = int(sys.argv[1])
N0 = 10000.
rho = 1e-9
theta = 2.5e-8
L = int(float(sys.argv[3]))

a = np.array([2.7, .2, 1.5, 2.7])
s = np.array([30000., 70000., 1.25e6 - 1e5, 1.]) / 25.0 / (2 * N0)
# a = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9.])[::-1]
# s = np.array([10000.0] * 9) / 25.0 / (2 * N0)
# a = np.array([1., .2, 5.0])
# s = np.array([10000., 2000., 10000.]) / 25.0 / (2 * N0)

a0 = a.copy()
s0 = s.copy()
true_parameters = (a, a, s)
print("true parameters")
print(np.array(true_parameters))
demography = psmcpp.scrm.demography_from_params((a * 2.0, a * 2.0, s))
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

import multiprocessing
p = multiprocessing.Pool(16)
obs_list = list(p.imap_unordered(perform_sim, 
    [(n, N0, theta, rho, L, demography, False, np.random.randint(0, sys.maxint))
        for _ in range(int(sys.argv[2]))]))
p.terminate()
p.join()
del p

if diagnostic:
# Extract true hidden states (1-2 coal time) from data
    ct = [(c1, _newick.tmrca(c2, "1", "2")) for c1, c2 in data[3]]
    cts = np.zeros(L)
    fs = frozenset([0,1])
    i = 0
    for span, tmrca in ct:
        cts[i:(i+span)] = tmrca
        i += span

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
# 
# obsfs_r = obsfs.sum(axis=1)
# obsfs_r[0] += obsfs_r[2]
# obsfs_r = obsfs_r[:2]

# 4. Optimize this function
hidden_states = np.array([0., 14.0]) / 25.0 / (2 * N0)
im = psmcpp._pypsmcpp.PyInferenceManager(n - 2, [obs_list[0][:10]], [0.0, 1.0],
        4.0 * N0 * theta, 4.0 * N0 * rho,
        block_size, 5, [0])
# prior = (a0,a0,s)
hs1 = im.balance_hidden_states(([1.], [1.], [1.]), 32)
# hs1 = np.concatenate((
#         np.linspace(0.0, 1.0, 15, False),
#         np.linspace(1.0, 15., 15, False),
#         [np.inf]))
print(hs1)
em = np.arange(3 *  (n - 1), dtype=int).reshape([3, n - 1])
em[0] = em[2] = 0
em[1] = 1
im = psmcpp._pypsmcpp.PyInferenceManager(n - 2, obs_list, hs1,
        4.0 * N0 * theta, 4.0 * N0 * rho,
        block_size, n, [0], em)

lam = 0.0
# s = hs1[1:-1] - hs1[:-2]
# a = np.array([35.34245,30.809847,24.796922,0.278533,18.540922,23.767104,24.88357,26.016034,29.116137,27.182875,
#             33.551099,35.028234,1.483683,50.104027,79.879866,3.321154,58.609216,93.80928,6.213876,100.,
#             30.892616,34.157583,18.468655,27.589934,25.300169,23.66569,25.767877,22.149063,27.295382,20.634101])
im.setParams((a0,a0,s0),False)
im.Estep()
ll_true = np.mean(im.loglik(lam))

s = np.array([2.5e4] * 4 + [1e5] * 15 + [1e6] * 2) / 25.0 / (2 * N0)
print(s)
K = len(s)
a = np.ones(K)
# a = np.array([2.273396,0.34565,0.21467,0.736181,3.716896,3.065967,2.64661,2.442381,2.498929,1.871542
#     ,1.777736,1.76801,1.778753,1.793984,2.132503,2.093779,1.824248,0.764588,10.])

if diagnostic:
    bk=im.block_keys()[0]
    cs = [sum(d.values()) for _, d in bk]
    xx = np.cumsum(cs)
    g = im.gammas()[0]
    g0 = np.zeros(g.shape)
    I = np.eye(g.shape[0])
    tcts = np.searchsorted(hs1, cts) - 1
    for i in range(g.shape[1] - 1):
        g0[:,i] = I[tcts[xx[i]]]

    import collections
    d = collections.defaultdict(lambda: collections.Counter())
    for i in range(g.shape[1]):
        d[tcts[xx[i] - 1]][frozenset(bk[i][1].items())] += 1
    im.setParams((a0,a0,s0),False)
    E = im.emission()
    F = np.zeros(E.shape)
    for k in d:
        for cc in d[k]:
            if sum(y for x,y in cc) == 10:
                for cc1, cc2 in cc:
                    F[k][cc1] += cc2 * d[k][cc]
    F /= F.sum(axis=1)[:,None]
    aoeu

# s = np.array([10000.] * 10) / 25.0 / N0
# K = len(s)
# x0 = np.ones([2, K]) * 20
# a = x0[0]
# b = x0[1]

im.seed = np.random.randint(0, sys.maxint)
llold = -np.inf
Qold = -np.inf

# print(im.Bs()[0][:,:6])
# print(im.alphas()[0][:,:6])
# print(im.gammas()[0][:,:6])

bounds = np.array([[0.1, 10.0]] * len(a))

class MyBounds(object):
    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmin = bool(np.all(x >= bounds[:, 0]))
        tmax = bool(np.all(x <= bounds[:, 1]))
        return tmax and tmin

def optimize_pg():
    def f(x):
        global s
        im.setParams((x, x, s), False)
        res = im.Q(lam)
        ret = -np.mean(res, axis=0)
        return ret
    def fprime(x):
        global s
        im.setParams((x, x, s), [(0, k) for k in range(K)])
        res = im.Q(lam)
        lls = np.array([ll for ll, jac in res])
        jacs = np.array([jac for ll, jac in res])
        #ret = -np.mean(jacs, axis=0)
        ret = (-np.mean(lls, axis=0), -np.mean(jacs, axis=0))
        print(ret[0])
        print(x)
        print(ret[1])
        print("------")
        return ret
    def project(x):
        return np.minimum(np.maximum(x, bounds[:,0]), bounds[:,1])
    x = a
    c = tau = 0.5
    f0, grad = fprime(x)
    p = -grad
    m = p.dot(grad)
    alpha = min([(bounds[i][int(p[i] > 0)] - x[i]) / p[i] for i in range(len(x))])
    f1 = f(project(x + alpha * p))
    while f1 > f0 + alpha * c * m:
        f1 = f(project(x + alpha * p))
        print(f0, f1)
        alpha *= tau
    return project(x + alpha * p)

def optimize_coord_ascent(c):
    print("Optimizing coordinate %d..." % c)
    # Coordinate-y ascent
    # c = i % K
    # print("Optimizing coordinate %d..." % c)
    # coords = [(i % 2, c), (i % 2, c)]
    # if c == 0:
    #     bounds = ((0.1, 100), (0.1 * a[c + 1], 10.0 * a[c + 1]))
    # elif c == K - 1:
    #     bounds = ((0.1 * b[c - 1], 10.0 * b[c - 1]), (0.1, 100.0))
    # else:
    #     bounds = ((0.1 * b[c - 1], 10.0 * b[c - 1]), (0.1 * a[c + 1], 10.0 * a[c + 1]))
    def fprime(x):
        global a, s
        # for coord, xx in zip(coords, x):
        # a0, b0 = smoothed_x(x)
        aa = a.copy()
        aa[c] = x
        im.setParams((aa, aa, s), [(0, c)])
        res = im.Q(lam)
        lls = np.array([ll for ll, jac in res])
        jacs = np.array([jac for ll, jac in res])
        #ret = -np.mean(jacs, axis=0)
        ret = (-np.mean(lls, axis=0), -np.mean(jacs, axis=0))
        print(x)
        print(ret[0])
        print(ret[1].reshape(3, K // 3))
        return ret
    def f(x):
        global a, s
        # for coord, xx in zip(coords, x):
        # a0, b0 = smoothed_x(x)
        aa = a.copy()
        aa[c] = x
        im.setParams((aa, aa, s), False)
        res = -np.mean(im.Q(lam))
        print(x, res)
        return res
    # res = scipy.optimize.fmin_l_bfgs_b(fprime, a[c], None, bounds=[tuple(bounds[c])], disp=False)
    res = scipy.optimize.minimize_scalar(f, bounds=bounds[c], method="bounded")
    return res.x

def optimize_fullgrad(iter):
    print("Optimizing all coordinates...")
    def fprime(x):
        global s
        im.setParams((x, x, s), [(0, k) for k in range(K)])
        res = im.Q(lam)
        lls = np.array([ll for ll, jac in res])
        jacs = np.array([jac for ll, jac in res])
        #ret = -np.mean(jacs, axis=0)
        ret = [-np.mean(lls, axis=0), -np.mean(jacs, axis=0)]

        # add penalty
        esfs, jac = psmcpp._pypsmcpp.sfs(n, (x, x, s), 0.0, np.inf, 4 * N0 * theta, True)
        diff = esfs[0, 0] - obsfs[0, 0]
        penalty = ALPHA_PENALTY * diff**2
        print('penalty', penalty, ret[0])
        ret[0] += penalty
        ret[1] += 2 * ALPHA_PENALTY * diff * jac[0, 0, :K]

        print(x)
        print(ret[0])
        print(ret[1])
        print("------")
        return ret
    # print('gradient check', scipy.optimize.check_grad(lambda x: fprime(x)[0], lambda x: fprime(x)[1], a))
    if iter <= 5:
        factr = 1e11
    elif 5 < iter <= 10:
        factr = 1e10
    else:
        factr = 1e9
    res = scipy.optimize.fmin_l_bfgs_b(fprime, a, None, bounds=[tuple(x) for x in bounds], disp=False, factr=factr)
    # print(res)
    return res[0]

i = 1
psmcpp._pypsmcpp.do_progress(False)
while i <= 20:
    if True or i % 5 == 0:
        a[:] = optimize_fullgrad(i)
    else:
        a[:] = optimize_pg()
    print("************** ITERATION %d ***************" % i)
    print(a)
    im.setParams((a, a, s), False)
    im.Estep()
    ll = np.mean(im.loglik(lam))
    print(" - Tru loglik:" + str(ll_true))
    print(" - New loglik:" + str(ll))
    print(" - Old loglik:" + str(llold))
    if ll < llold:
        print("*** Log-likelihood decreased")
    llold = ll
    esfs = psmcpp._pypsmcpp.sfs(n, (a,a,s), 0.0, 14.0, 4 * N0 * theta, False)
    print("D(Ob. sfs || Est. sfs):", kl(obsfs, esfs))
    print("D(Ob. sfs || True sfs):", kl(obsfs, esfs0))
    print(esfs)
    i += 1
