from __future__ import absolute_import, division, print_function
import numpy as np
import sys

from . import _smcpp


def _TDMASolve(a, b, c, d):
    # a, b, c == diag(-1, 0, 1)
    n = len(d)  # n is the numbers of rows, a and c has length n-1
    for i in xrange(n - 1):
        d[i + 1] -= 1. * d[i] * a[i] / b[i]
        b[i + 1] -= 1. * c[i] * a[i] / b[i]
    for i in reversed(xrange(n - 1)):
        d[i] -= d[i + 1] * c[i] / b[i + 1]
    return [d[i] / b[i] for i in xrange(n)]


class CubicSpline:

    def __init__(self, x, y):
        x = np.array(x)
        y = np.array(y)
        self._x = x
        h = x[1:] - x[:-1]
        j = y[1:] - y[:-1]
        # Subdiagonal
        a = h[:-1] / 3.
        a = np.append(a, h[-1])
        # Diagonal
        b = (h[1:] + h[:-1]) / 3.
        b = 2. * np.concatenate([[h[0]], b, [h[-1]]])
        # Superdiagonal
        c = h[1:] / 3.
        c = np.concatenate([[h[0]], c])
        # RHS
        jh = j / h
        d = np.concatenate([[3 * jh[0]], jh[1:] - jh[:-1], [-3. * jh[-1]]])
        # Solve tridiagonal system
        cb = np.array(_TDMASolve(a, b, c, d), dtype=object)
        ca = (cb[1:] - cb[:-1]) / h / 3.
        ca = np.append(ca, 0.0)
        cc = jh - h * (2. * cb[:-1] + cb[1:]) / 3.
        cc = np.append(cc, 3. * ca[-2] * h[1]**2 + 2 * cb[-2] * h[-1] + cc[-1])
        self._coef = np.array([ca, cb, cc, y])
        if self._x[0] > 0:
            self._x = np.concatenate([[0], self._x])
            self._coef = np.insert(self._coef, 0, [0, 0, 0, y[0]], axis=1)
        # if self._x[-1] < _smcpp.T_MAX:
        #     self._x = np.concatenate([self._x, [_smcpp.T_MAX]])
        #     self._coef = np.insert(self._coef, -1, [0, 0, 0, y[-1]], axis=1)

    def roughness(self):
        # Integral of squared second derivative
        a, b = self._coef[:2, :-1]
        xi = np.diff(self._x)
        return (12 * a**2 * xi**3 + 12 * a * b * xi**2 + 4 * b**2 * xi).sum()

    def eval(self, points):
        points = np.atleast_1d(points)
        ip = np.maximum(0, np.searchsorted(self._x, points) - 1)
        exp = np.arange(4)[::-1, None]
        xi = (points - self._x[ip])**exp
        return (self._coef[:, ip] * xi).sum(axis=0)

    def dump(self):
        s = "Piecewise[{"
        arr = []
        for k in range(len(self._x) - 1):
            u = "(x-(%.2f))" % self._x[k]
            arr.append("{" + "+".join(
                "%.6f*%s^%d" % (float(xi), u, 3 - i) 
                for i, xi in enumerate(self._coef[:, k])) + ",x>=%.2f&&x<%.2f}" % tuple(self._x[k:k + 2]))
        s += ",\n".join(arr) + "}];"
        print(s, file=sys.stderr)
