import numpy as np
import numbers
import scipy.interpolate
import math
import sys


def _align(p1, p2):
    '''
    Given PPoly's :p1:, :p2:, return tuple :(q1, q2): such that q1 and
    q2 have the same breakpoints and represent p1 and p2, respectively.

    It's assumed that the first and last breakpoints of p1 and p2 are 
    equal.
    '''
    nx = np.array(sorted(frozenset(list(p1.x) + list(p2.x))))
    ret = []
    for poly in (p1, p2):
        c = np.zeros([max(p1.c.shape[0], p2.c.shape[0]), len(nx) - 1])
        pieces = np.searchsorted(poly.x, nx[:-1], side="right") - 1
        for i in range(c.shape[0]):
            for j, ind in enumerate(pieces):
                x = nx[j] - poly.x[ind]
                pv = np.polyval(np.polyder(poly.c[:, ind], i), x)
                c[c.shape[0] - i - 1, j] = pv / math.factorial(i)
        ret.append(PPoly(c, nx))
    return tuple(ret)


class PPoly(scipy.interpolate.PPoly):

    def __sub__(self, other):
        p1, p2 = _align(self, other)
        return PPoly(p1.c - p2.c, p1.x)

    def __add__(self, other):
        p1, p2 = _align(self, other)
        return PPoly(p1.c + p2.c, p1.x)

    def __mul__(self, other):
        if not isinstance(other, PPoly):
            # This should hopefully be a constant or something.
            return PPoly(self.c * other, self.x)
        p1, p2 = _align(self, other)
        cs = [np.polymul(c1, c2) for c1, c2 in zip(p1.c.T, p2.c.T)]
        c = np.zeros([max([len(c) for c in cs]), p1.c.shape[1]])
        for i, cc in enumerate(cs):
            c[-len(cc):, i] = cc
        return PPoly(c, p1.x)

    def to_s(self):
        'Return a Mathematica representation of this PPoly.'
        s = "Piecewise[{\n"
        arr = []
        order = self.c.shape[0] - 1
        for k in range(len(self.x) - 1):
            u = "(x-(%.2f))" % self.x[k]
            arr.append("{" + "+".join(
                "%.6f*%s^%d" % (float(xi), u, order - i) 
                for i, xi in enumerate(self.c[:, k])) + ",x>=%.2f&&x<%.2f}" % tuple(self.x[k:k + 2]))
        s += ",\n".join(arr) + "\n}];"
        return s


def _bspline_basis(t, n):
    '''Bspline basis for knot sequence :t: up to order :n:.'''
    key = (tuple(t), n)
    if key not in _bspline_basis.memo:
        memo = {}
        def B(i, k):
            if k == 0:
                return PPoly([[0., 1., 0.]], [t[0], t[i], t[i + 1], t[-1]])
            if (i, k) not in memo:
                p1 = B(i, k - 1)
                f1 = PPoly(
                        [[0., 1. / (t[i + k] - t[i]), 0.], [0, 0, 0]],
                        [t[0], t[i], t[i + k], t[-1]]
                        )
                p2 = B(i + 1, k - 1)
                f2 = PPoly(
                        [[0., 1. / (t[i + 1] - t[i + k + 1]), 0.], [0., 1., 0.]],
                        [t[0], t[i + 1], t[i + k + 1], t[-1]]
                        )
                memo[(i, k)] = p1 * f1 + p2 * f2
            return memo[(i, k)]
        _bspline_basis.memo[key] = [B(i, n) for i in range(len(t) - n - 1)]
    return _bspline_basis.memo[key]
_bspline_basis.memo = {}


class BSpline(CubicSpline):
    def _fit(self):
        b = _bspline_basis(self._x, 3)
        zero = PPoly([[0.]], [x[0], x[-1]])
        poly = sum([bb * yy for yy, bb in zip(y, b)], zero)
        self._coef = poly.c
