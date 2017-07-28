'Beta kernel density estimation'
import numpy as np
import scipy.stats
import scipy.optimize
import scipy.special
import scipy.interpolate


from . import _smcpp


EULER_GAMMA = 0.577215664901532860


def harmonic_number(x):
    'Generalized harmonic number'
    return scipy.special.digamma(1. + x) + EULER_GAMMA


def quantile(X, h, q):
    # def g(y):
    #     return scipy.stats.beta.pdf(X[None, :], 1. + y / h, 1. + (1. - y) / h).mean(axis=1)
    x = np.linspace(0, 1., 10000)[1:]
    y = _smcpp.beta_de_avg_pdf(X, x, h)
    x = np.r_[0, x]
    y = np.cumsum(np.r_[0, y])
    y /= y[-1]
    F = scipy.interpolate.interp1d(x, y)
    return np.array([scipy.optimize.brentq(lambda x: F(x) - qq, 0., 1., rtol=.01)
                     for qq in q])


def positive_part(f, a, b):
    '''For f which monotonically increases on [a, b], return 
       [l, r] \subset [a, b] such that f(x) >= 0 for all x \in [l, r].
       Return None for empty set.
    '''
    assert f(b) >= 0
    if f(a) >= 0:
        l = a
    else:
        l = scipy.optimize.brentq(f, a, b)
    return [l, b]


def sample_beta_kernel(X, mu, h):
    '''
    Draw Poisson(mu) samples from Beta(X, 1 + y / h, 1 + (1 - y) / h).
    '''
    M = np.random.poisson(mu)
    if X in [0, 1]:
        return np.full(M, X)

    def g(y):
        return scipy.stats.beta.logpdf(X, 1. + y / h, 1. + (1. - y) / h)
    def dg(y):
        '(Sign of) dg/dy'
        return (harmonic_number((1 - y) / h) - 
                harmonic_number(y / h) - 
                np.log(1 / X - 1))

    # Slice sample from unimodal density g.
    if dg(0) < 0:  # g is monotone decreasing
        def sl(z):
            l, r = positive_part(lambda y: g(1. - y) - z, 0, 1)
            return [1 - r, 1 - l]
    elif dg(1) > 0:
        def sl(z):
            return positive_part(lambda y: g(y) - z, 0, 1)
    else:
        assert dg(0) > 0 and dg(1) < 0
        mid = scipy.optimize.brentq(dg, 0, 1)
        def sl(z):
            l, _ = positive_part(lambda y: g(y) - z, 0, mid)
            l1, _ = positive_part(lambda y: g(1 - y) - z, 0, 1 - mid)
            return [l, 1 - l1]
    x0 = 0.5  # initial value is discarded
    ret = np.zeros(M)
    for i in range(M):
        z = g(x0) - np.random.exponential()
        l, r = sl(z)
        x0 = ret[i] = np.random.uniform(l, r)
    return ret
