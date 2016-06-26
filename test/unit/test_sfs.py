import numpy as np
np.set_printoptions(linewidth=120)
import sys
import ad

import smcpp._smcpp, smcpp.model


def test_human():
    n = 10
    hs = [0, 0.002, 0.0024992427075529156, 0.0031231070556282147, 0.0039027012668429368, 0.0048768988404573679, 0.0060942769312431738, 0.0076155385891087321, 0.0095165396414589112, 0.011892071150027212, 0.014860586049702963, 0.018570105657341358, 0.023205600571298769, 0.028998214001102113, 0.036236787437156658, 0.045282263373729439, 0.0565856832591419, 0.070710678118654752, 0.088361573317084705, 0.11041850887031313, 0.1379813265364985, 0.15879323234898887, 0.22690003122622915, 0.28675095012241164, 0.35039604900931776, 0.4174620285802807,
          0.48093344839252727, 0.54048403452772453, 0.58902987679112695, 0.63973400753929655, 0.6661845719884536, 0.68097444812291441, 0.69652310395210704, 0.71291262669986732, 0.73023918985303526, 0.74861647270557707, 0.76818018497781393, 0.7890941548490632, 0.81155867710242946, 0.8429182938518559, 0.88146343535942318, 0.92368486081866963, 0.97035848127888702, 1.0225351498208244, 1.1293598575982273, 1.2553186915845398, 1.468142830257521, 1.7982719448467761, 2.3740247153419043, 3.2719144602927757, 4.8068671176749671, 49.899999999999999]
    s = [0.002, 0.00049924270755291556, 0.00062386434807529907, 0.00077959421121472213, 0.00097419757361443156, 0.0012173780907858058, 0.0015212616578655584, 0.0019010010523501783, 0.0023755315085683005, 0.0029685148996757508, 0.0037095196076383959, 0.0046354949139574102, 0.0057926134298033442, 0.0072385734360545448, 0.0090454759365727819, 0.011303419885412461,
         0.014124994859512852, 0.017650895198429953, 0.022056935553228421, 0.027562817666185374, 0.034443085525912243, 0.043040815163128798, 0.0537847217117913, 0.067210536757978667, 0.083987721931547688, 0.10495285078070138, 0.13115132347527858, 0.16388949439075173, 0.20479981185031038, 0.25592221813754867, 0.31980586869051764, 0.39963624257870101, 0.49939398246933342]
    a = np.array([ad.adnumber(x) for x in a])
    m = smcpp.model.SMCModel(s, [0])
    m.x[0] = m.x[1] = a
    for i in range(1, len(hs)):
        t0 = hs[i - 1]
        t1 = hs[i]
        sfs = smcpp._smcpp.raw_sfs(m, n, t0, t1, jacobian=True)
        eps = 1e-8
        for k in range(1):  # range(m.K)
            ac = a.copy()
            ac[k] += eps
            m.x[0] = m.x[1] = ac
            sfs2 = smcpp._smcpp.raw_sfs(m, n, t0, t1, jacobian=False)
            # for j in range(n - 1):
            for j in range(1):
                jaca = sfs[0, j].d(a[k])
                j1 = sfs2[0, j]
                j2 = sfs[0, j] + eps * jaca
                jacb = (sfs2[0, j] - sfs[0, j]) / eps
                print(k, j, sfs[0, j], sfs2[0, j], jaca, jacb)
                # assert abs(j1 - j2) < eps
    assert False

def test_d():
    a = d['a']
    b = d['b']
    s = d['s']
    K = len(a)
    # a = np.ones(10) * 1.0
    # b = 2 * a
    # s = np.array([0.1] * K)
    n = 10
    t0 = 0.0
    t1 = 14.0
    coords = [(aa, bb) for aa in [0, 1] for bb in range(K)]
    sfs, jac = _pypsmcpp.sfs(n, (a, b, s), t0, t1, 4 * N0 * theta, coords)
    jac.shape = (3, n - 1, 2, K)
    eps = 1e-8
    for ind in (0, 1):
        for k in range(K):
            args = [a, b, s]
            args[ind] = args[ind] + eps * I[k]
            # print(args)
            la, lb, ls = args
            sfs2 = _pypsmcpp.sfs(n, (la, lb, ls), t0, t1, 4 * N0 * theta, False)
            for i in (0, 1, 2):
                for j in range(n - 1):
                    jaca = jac[i, j, ind, k]
                    j1 = sfs2[i, j]
                    j2 = sfs[i, j] + eps * jaca
                    print(ind, k, i, j, sfs[i,j], sfs2[i,j], jaca, (sfs2[i,j] - sfs[i,j]) / eps)
                    # assert abs(j1 - j2) < eps
    assert False

def test_fail3():
    t0 = 0.0
    t1 = np.inf
    n = 2
    a, b = np.array(
        [[2.785481, 2.357832, 3.189695, 4.396804, 6.340599, 6.74133, 10., 10., 10., 10., 10., 10., 10., 10., 0.1],
         [2.702762, 2.313371, 3.073819, 4.199494, 7.082639, 7.427921, 9.92296, 10., 10., 10., 10., 10., 10., 10., 1.1]])
    s = [0.001, 0.000733, 0.00127, 0.0022, 0.003812, 0.006606, 0.011447, 0.019836, 0.034371, 0.059557, 0.103199, 0.178822,
            0.30986, 0.536921, 0.930368]
    sfs, jac = _pypsmcpp.sfs(n, (a, b, s), t0, t1, 4 * N0 * theta, jacobian=True)


def test_matching_diff_nodiff(demo):
    from timeit import default_timer as timer
    a, b, s = demo
    n = 5
    start = timer()
    seed = np.random.randint(0, 10000000)
    sfs1, rsfs1, jac = _pypsmcpp.sfs((a, b, s), n, num_samples, 0., np.inf, NTHREAD, 1.0, jacobian=True, seed=seed)
    end = timer()
    print("Run-time with jacobian: %f" % (end - start))
    start = timer()
    sfs2, rsfs2 = _pypsmcpp.sfs((a, b, s), S, M, 0., np.inf, NTHREAD, 1.0, jacobian=False, seed=seed)
    end = timer()
    print("Run-time without: %f" % (end - start))
    print(sfs1)
    print(sfs2)
    print(jac)

def test_correct_const(constant_demo_1, constant_demo_1000):
    # Make as close to constant as possible
    obs = [np.array([[1, 0, 0], [1, 0, 0]], dtype=np.int32)]
    for d, mult in ((constant_demo_1, 1.0), (constant_demo_1000, 1000.0)):
        for n in (2, 3, 10, 20):
            sfs, rsfs, jac = _pypsmcpp.sfs((a, b, s), n, S, M, 0., np.inf, NTHREAD, 1.0, jacobian=True, seed=seed)
            im = _pypsmcpp.PyInferenceManager(n - 2, obs, np.array([0.0, np.inf]), 1e-8, 1e-8, 1, NTHREAD, num_samples)
            sfs = im.sfs(d, 0, np.inf)
            rsfs = _pypsmcpp.reduced_sfs(sfs) 
            print(rsfs)
            rsfs /= rsfs[1:].sum()
            print(rsfs)
            for k in range(1, n):
                expected = (1. / k) / sum([1. / j for j in range(1, n)])
                assert (rsfs[k] - expected) / expected < 1e-2

def test_fail():
    t0 = 0.0
    t1 = np.inf
    n = 100
    a, b = [[10., 10., 10., 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 6.044083],
            [10., 10., 10., 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.1]]
    s = [0.058647, 0.093042, 0.147608, 0.234176, 0.371514, 0.589396, 0.93506, 1.483445, 2.353443, 3.733669]
    sfs, jac = _pypsmcpp.sfs(n, (a, b, s), t0, t1, 4 * N0 * theta, jacobian=True)


def test_fail2():
    t0 = 0.0
    t1 = np.inf
    n = 100
    a = [10., 10., 10., 10., 0.1, 0.1, 0.1, 4.141793, 10., 10.]
    b = [10., 10., 10., 4.513333, 0.1, 0.1, 0.1, 2.601576, 4.991452, 1.1]
    s = [0.058647, 0.093042, 0.147608, 0.234176, 0.371514, 0.589396, 0.93506, 1.483445, 2.353443, 3.733669]
    sfs, jac = _pypsmcpp.sfs(n, (a, b, s), t0, t1, 4 * N0 * theta, jacobian=True)
