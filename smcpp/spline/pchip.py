from __future__ import absolute_import, division, print_function
import numpy as np

from .common import smooth_abs
from .cubic import CubicSpline


class PChipSpline(CubicSpline):
    'A C1 monotone spline.'
    def _pchipendpoint(self, h1, h2, del1, del2):
        d = ((2 * h1 + h2) * del1 - h1 * del2) / (h1 + h2)
        if np.sign(d) != np.sign(del1):
            d = 0
        elif (np.sign(del1) != np.sign(del2)) and (smooth_abs(d) > smooth_abs(3 * del1)):
            d = 3 * del1
        return d


    def _pchipslopes(self, h, delta):
        n = len(h) + 1
        d = np.zeros_like(delta)
        k = np.where(np.sign(delta[:(n - 2)]) *
                     np.sign(delta[1:(n + 1)]) > 0)[0] + 1
        w1 = 2 * h[k] + h[k - 1]
        w2 = h[k] + 2 * h[k - 1]
        d[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])
        d[0] = self._pchipendpoint(h[0], h[1], delta[0], delta[1])
        d = np.append(d, self._pchipendpoint(
            h[n - 2], h[n - 3], delta[n - 2], delta[n - 3]))
        return d


    def _fit(self):
        x = self._x
        y = self._y
        h = np.diff(x)
        delta = np.diff(y) / h
        d = self._pchipslopes(h, delta)
        n = len(x)
        c = (3 * delta - 2 * d[:n - 1] - d[1:n]) / h
        b = (d[:n - 1] - 2 * delta + d[1:n]) / h**2
        b = np.concatenate([b, [0.0]])
        c = np.concatenate([c, [0.0]])
        self._coef = np.array([b, c, d, y])

