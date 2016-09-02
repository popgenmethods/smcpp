import pytest
import ad
import numpy as np

import smcpp
from smcpp import util
from smcpp.jcsfs import JointCSFS
from smcpp.model import SMCModel, SMCTwoPopulationModel, PiecewiseModel

@pytest.fixture
def model1():
    s = np.diff(np.logspace(np.log10(.001), np.log10(4), 4))
    ret = SMCModel(s, np.logspace(np.log10(.001), np.log10(4), 5))
    ret[:] = np.log(np.arange(1, 6)[::-1])
    ret[:] = 1.0
    s = [0.5, 1.0]
    a = [1.0, 4.]
    ret = PiecewiseModel(a, s)
    return ret

@pytest.fixture
def model2():
    s = np.diff(np.logspace(np.log10(.001), np.log10(4), 4))
    ret = SMCModel(s, np.logspace(np.log10(.001), np.log10(4), 5))
    ret[:] = np.log(np.arange(1, 6))
    ret[:] = 1.0
    s = [.1, .2, .3]
    a = [2.0, 4.0, 2.]
    ret = PiecewiseModel(a, s)
    return ret

@pytest.fixture
def model12(model1, model2):
    return SMCTwoPopulationModel(model1, model2, .5)

@pytest.fixture
def jcsfs():
    return JointCSFS(5, 2, 2, 0, [0.0, 0.5, 1.0, np.inf])

def _cumsum0(x):
    return np.concatenate([[0.], np.cumsum(x)])

def _concat_models(model1, model2, split):
    # Return a model which is model1 before split and model2 after it.
    ary = []
    for m in (model1, model2):
        a = m.stepwise_values()
        cs = _cumsum0(m.s)
        cs[-1] = np.inf
        ip = np.searchsorted(cs, split)
        cs = np.insert(cs, ip, split)
        sp = np.diff(cs)
        ap = np.insert(a, ip, a[ip - 1])
        sp[-1] = 1.
        ary.append((sp, ap, ip))
    s, a = [np.concatenate([ary[1][i][:ary[1][2]], ary[0][i][ary[0][2]:]]) for i in [0, 1]]
    return PiecewiseModel(a, s)
np.set_printoptions(precision=3, linewidth=100)

def test_d(model12):
    ts = [0.0, 0.5, 1.0, np.inf]
    n1 = 10 
    n2 = 8
    model1 = model12.model1
    ders = model1.a = np.array([ad.adnumber(x, tag=i) for i, x in enumerate(model1.a)], dtype=object)
    j0 = smcpp._smcpp.joint_csfs(n1, n2, 2, 0, model12, [0., 1.0], 100)[0]
    for i in range(2):
        model1.a[i].x += 1e-8
        j1 = np.array(smcpp._smcpp.joint_csfs(n1, n2, 2, 0, model12, [0., 1.0], 100))[0]
        model1.a[i].x -= 1e-8
        for x, y in zip(j0.flat, j1.flat):
            print(x.d(ders[i]), float(y - x) * 1e8)
    assert False
    

def test_marginal_pop1(model1, model2):
    ts = [0., 1., 2., np.inf]
    n1 = 5
    n2 = 10
    j = JointCSFS(n1, n2, 2, 0, ts, 1000)
    for split in [0.1, 0.5, 1., 1.5, 2.5]:
        jc = j.compute(model1, model2, split)
        for t1, t2, jjc in zip(ts[:-1], ts[1:], jc):
            A1 = smcpp._smcpp.raw_sfs(model1, n1, t1, t2).astype('float')
            A2 = jjc.sum(axis=(-1, -2)).astype('float')
            assert np.allclose(A1.flat[1:-1], A2.flat[1:-1], 1e-1, 0)

def test_marginal_pop2(model1, model2):
    n1 = 8
    n2 = 10
    j = JointCSFS(n1, n2, 2, 0, [0.0, np.inf], 100)
    for split in [0.1, 0.25, 0.5, 0.75, 1.0, 2.0]:
        true_model2 = _concat_models(model1, model2, split)
        csfs = smcpp._smcpp.raw_sfs(true_model2, n2 - 2, 0., np.inf)
        A1 = util.undistinguished_sfs(csfs).astype('float')[1:]
        jc = j.compute(model1, model2, split)[0]
        A2 = jc.sum(axis=(0, 1, 2)).astype('float')[1:-1]
        assert np.allclose(A1, A2, 1e-1, 0)


def test_equal_py_c(model1, model2):
    ts = [0., 0.5, 1., 2., np.inf]
    n1 = 7
    n2 = 5
    py_jcsfs = JointCSFS(n1, n2, 2, 0, ts, 1000)
    model = smcpp.model.SMCTwoPopulationModel(model1, model2, 0.)
    for split in ts[1:-1]:
        j1 = py_jcsfs.compute(model1, model2, split)
        model.split = split
        j2 = np.array(smcpp._smcpp.joint_csfs(n1, n2, 2, 0, model, ts, 1000)).astype('float')
        assert np.allclose(j1, j2, 1e-1, 0)

def _model_to_momi_events(s, a, pop):
    sp = np.concatenate([[0.], s])[:-1]
    return [("-en", tt, pop, aa) for tt, aa in zip(sp, a.astype('float'))]

def test_vs_momi_together(model1, model2):
    import momi, momi.demography
    n1 = 7
    n2 = 5
    split = .02

    # Conditioned version
    model = smcpp.model.SMCTwoPopulationModel(model1, model2, split)
    jc = np.array(smcpp._smcpp.joint_csfs(n1, n2, 2, 0, model, [0., np.inf], 1000)).astype('float')[0]
    jm = np.zeros([n1 + 3, n2 + 1])
    for a1 in range(3):
        for b1 in range(n1 + 1):
            for a2 in range(1):
                for b2 in range(n2 + 1):
                    jm[a1 + b1, a2 + b2] += jc[a1, b1, a2, b2]
    np.testing.assert_allclose(jm.sum(), jc.sum())
    print("")
    print(jm)
    print(jm.sum(axis=0))
    print(jm.sum(axis=1))

    # Momi version
    m2s = _cumsum0(model2.s)
    tm2 = np.searchsorted(m2s, split)
    m2s = m2s[:tm2]
    m2a = model2.stepwise_values()[:tm2]
    p2_events = [("-en", tt, "pop2", aa * 2.) for tt, aa in zip(m2s, m2a)]
    events = _model_to_momi_events(model1.s, model1.stepwise_values() * 2., "pop1")
    events.append(("-ej", split, "pop2", "pop1"))
    events += p2_events
    demo = momi.demography.make_demography(events, ["pop1", "pop2"], (n1 + 2, n2))
    configs = [((n1 + 2 - a, a), (n2 - b, b))
               for a in range(n1 + 3) for b in range(n2 + 1)
               if 0 < a + b < n1 + n2 + 2]
    mconf = momi.config_array(("pop1", "pop2"), configs)
    esfs = momi.expected_sfs(demo, mconf, mut_rate=1.0)
    j_momi = np.zeros_like(jm)
    for ((_, a), (_, b)), x in zip(configs, esfs):
        j_momi[a, b] += x
    print(j_momi)
    print(j_momi.sum(axis=0))
    print(j_momi.sum(axis=1))

def test_vs_momi_apart(model1, model2):
    import momi, momi.demography
    n1 = 15
    n2 = 17
    split = .1

    # Conditioned version
    model = smcpp.model.SMCTwoPopulationModel(model1, model2, split)
    jc = np.array(smcpp._smcpp.joint_csfs(n1, n2, 1, 1, model, [0., np.inf], 1000)).astype('float')[0]
    jm = np.zeros([n1 + 2, n2 + 2])
    for a1 in range(2):
        for b1 in range(n1 + 1):
            for a2 in range(2):
                for b2 in range(n2 + 1):
                    jm[a1 + b1, a2 + b2] += jc[a1, b1, a2, b2]
    np.testing.assert_allclose(jm.sum(), jc.sum())
    print("")
    print(jm)

    # Momi version
    m2s = _cumsum0(model2.s)
    tm2 = np.searchsorted(m2s, split)
    m2s = m2s[:tm2]
    m2a = model2.stepwise_values()[:tm2]
    p2_events = [("-en", tt, "pop2", aa * 2.) for tt, aa in zip(m2s, m2a)]
    events = _model_to_momi_events(model1.s, model1.stepwise_values() * 2., "pop1")
    events.append(("-ej", split, "pop2", "pop1"))
    events += p2_events
    demo = momi.demography.make_demography(events, ["pop1", "pop2"], (n1 + 1, n2 + 1))
    configs = [((n1 + 1 - a, a), (n2 + 1 - b, b))
               for a in range(n1 + 2)
               for b in range(n2 + 2)
               if 0 < a + b < n1 + n2 + 2]
    mconf = momi.config_array(("pop1", "pop2"), configs)
    esfs = momi.expected_sfs(demo, mconf, mut_rate=1.0)
    j_momi = np.zeros_like(jm)
    for ((_, a), (_, b)), x in zip(configs, esfs):
        j_momi[a, b] += x
    print(j_momi)
    # np.testing.assert_allclose(j_momi, jm, rtol=1e-3)

def test_bug(model1, model2):
    n1 = 10
    n2 = 10
    split = 0.02

    # Conditioned version
    model = smcpp.model.SMCTwoPopulationModel(model1, model2, split)
    jc = np.array(smcpp._smcpp.joint_csfs(n1, n2, 1, 1, model, [0., np.inf], 1000)).astype('float')[0]
    jm = np.zeros([n1 + 2, n2 + 2])
    for a1 in range(2):
        for b1 in range(n1 + 1):
            for a2 in range(2):
                for b2 in range(n2 + 1):
                    jm[a1 + b1, a2 + b2] += jc[a1, b1, a2, b2]
    np.testing.assert_allclose(jm.sum(), jc.sum())
    print("")
    print(jm)

    # Momi version
    m2s = _cumsum0(model2.s)
    tm2 = np.searchsorted(m2s, split)
    m2s = m2s[:tm2]
    m2a = model2.stepwise_values()[:tm2]
    p2_events = [("-en", tt, "pop2", aa * 2.) for tt, aa in zip(m2s, m2a)]
    events = _model_to_momi_events(model1.s, model1.stepwise_values() * 2., "pop1")
    events.append(("-ej", split, "pop2", "pop1"))
    events += p2_events
    demo = momi.demography.make_demography(events, ["pop1", "pop2"], (n1 + 1, n2 + 1))
    configs = [((n1 + 1 - a, a), (n2 + 1 - b, b))
               for a in range(n1 + 2)
               for b in range(n2 + 2)
               if 0 < a + b < n1 + n2 + 2]
    mconf = momi.config_array(("pop1", "pop2"), configs)
    esfs = momi.expected_sfs(demo, mconf, mut_rate=1.0)
    j_momi = np.zeros_like(jm)
    for ((_, a), (_, b)), x in zip(configs, esfs):
        j_momi[a, b] += x
    print(j_momi)
    # np.testing.assert_allclose(j_momi, jm, rtol=1e-3)
# def test_jcsfs(jcsfs, model1, model2):
#     jcsfs.compute(model1, model2, 0.25)
# def test_jcsfs(jcsfs, model1, model2):
#     jcsfs.compute(model1, model2, 0.25)
