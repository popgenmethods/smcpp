from __future__ import absolute_import, division, print_function
import numpy as np
import re
import sys
import wrapt

from .common import smooth_abs, polyval
from .spline import Spline

def _TDMASolve(a, b, c, d):
    # a, b, c == diag(-1, 0, 1)
    n = len(d)  # n is the numbers of rows, a and c has length n-1
    for i in range(n - 1):
        d[i + 1] -= 1. * d[i] * a[i] / b[i]
        b[i + 1] -= 1. * c[i] * a[i] / b[i]
    for i in reversed(range(n - 1)):
        d[i] -= d[i + 1] * c[i] / b[i + 1]
    return [d[i] / b[i] for i in range(n)]


class CubicSpline(Spline):
    'A spline.'
    def __init__(self, x, y=None):
        if y is None:
            y = np.zeros_like(x, dtype='object')
        Spline.__init__(self, 3, x, y)


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

    def tv(self):
        'Integral of absolute value of first derivative.'
        s = 0.
        for c, x1, x2 in zip(self._coef.T[:-1], self._x[:-2], self._x[1:-1]):
            x = np.linspace(x1, x2, 50)
            d1 = polyval(np.polyder(c, 1), x)
            y = smooth_abs(d1)
            s += np.trapz(y, x)
        return s

    def roughness(self):
        'Integral of squared second derivative.'
        a, b = self._coef[:2, :-1]
        xi = np.diff(self._x)
        return (12 * a**2 * xi**3 + 12 * a * b * xi**2 + 4 * b**2 * xi).sum()


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
