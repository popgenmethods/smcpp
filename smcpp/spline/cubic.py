from __future__ import absolute_import, division, print_function
import numpy as np
import re
import sys

from .common import smooth_abs, polyval


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
    'A C2 spline.'
    def __init__(self, x, y):
        self._x = np.array(x)
        self._y = np.array(y)
        self._fit()
        # Flat outside of boundaries
        if self._x[0] > 0:
            self._x = np.concatenate([[0], self._x])
            self._coef = np.insert(self._coef, 0, [0, 0, 0, y[0]], axis=1)
        self._x = np.append(self._x, np.inf)
        self._coef = np.insert(self._coef, -1, [0, 0, 0, y[-1]], axis=1)


    def _fit(self):
        x = self._x
        y = self._y
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

    def roughness(self):
        'Integral of squared second derivative.'
        a, b = self._coef[:2, :-1]
        xi = np.diff(self._x)
        return (12 * a**2 * xi**3 + 12 * a * b * xi**2 + 4 * b**2 * xi).sum()

    def integrated_curvature(self):
        '''Integral of curvature:

            \int |f''| / (1 + (f')^2)^(3/2)

        This cannot be evaluated analytically, so the integral is
        numeric.
        '''
        s = 0.
        for c, x1, x2 in zip(self._coef.T[:-1], self._x[:-2], self._x[1:-1]):
            x = np.linspace(x1, x2, 50)
            d2 = polyval(np.polyder(c, 2), x)
            d1 = polyval(np.polyder(c, 1), x)
            y = smooth_abs(d2) * (1. + d1**2)**(-1.5)
            s += np.trapz(y, x)
        return s

    def squared_d2(self):
        '''Array of numpy.poly1d representing the squared second
        derivative of self.
        '''
        d2 = [np.poly1d(y).deriv(2)**2 for y in self._coef.T]
        return (self._x, d2)

    def __call__(self, points):
        'Evaluate at points.'
        points = np.atleast_1d(points)
        ip = np.searchsorted(self._x, points) - 1
        exp = np.arange(4)[::-1, None]
        xi = (points - self._x[ip])**exp
        ret = (self._coef[:, ip] * xi).sum(axis=0)
        # The spline is constrained to be flat outside of the knot range
        ret[ip < 0] = self._coef[-1, 0]
        return ret

    def dump(self, file=sys.stderr):
        s = "Piecewise[{"
        arr = []
        for k in range(len(self._x) - 1):
            u = "(x-(%e))" % self._x[k]
            arr.append("{" + "+".join(
                "%e*%s^%d" % (float(xi), u, 3 - i)
                for i, xi in enumerate(self._coef[:, k])) + ",x>=%e&&x<%e}" % tuple(self._x[k:k + 2]))
        s += ",\n".join(arr) + "}];"
        # convert to mathematica style
        s = re.sub(r'e([+-]\d+)', r'*^\1', s)
        print(s, file=file)
