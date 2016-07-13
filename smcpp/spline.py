from __future__ import absolute_import, division, print_function
import numpy as np
import sys
import ad.admath

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
        self._x = np.array(x)
        self._y = np.array(y)
        self._fit()
        if self._x[0] > 0:
            self._x = np.concatenate([[0], self._x])
            self._coef = np.insert(self._coef, 0, [0, 0, 0, y[0]], axis=1)
        # if self._x[-1] < _smcpp.T_MAX:
        #     self._x = np.concatenate([self._x, [_smcpp.T_MAX]])
        #     self._coef = np.insert(self._coef, -1, [0, 0, 0, y[-1]], axis=1)


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


@np.vectorize
def _smooth_abs(x):
    return (x**2 + 1e-3)**0.5


class AkimaSpline(CubicSpline):
# http://www.lfd.uci.edu/~gohlke/code/akima.py.html
# Copyright (c) 2007-2015, Christoph Gohlke
# Copyright (c) 2007-2015, The Regents of the University of California
# Produced at the Laboratory for Fluorescence Dynamics
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the copyright holders nor the names of any
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
    def _fit(self):
        x = self._x
        y = self._y
        dx = np.diff(x)
        n = len(x)
        m = np.diff(y) / dx
        mm = 2.0 * m[0] - m[1]
        mmm = 2.0 * mm - m[0]
        mp = 2.0 * m[n - 2] - m[n - 3]
        mpp = 2.0 * mp - m[n - 2]
        m1 = np.concatenate(([mmm], [mm], m, [mp], [mpp]))
        dm = _smooth_abs(np.diff(m1))
        f1 = dm[2:n + 2]
        f2 = dm[0:n]
        f12 = f1 + f2
        ids = f12 > 1e-9 * np.max(f12)
        b = m1[1:n + 1]
        b[ids] = (f1[ids] * m1[ids + 1] + f2[ids] * m1[ids + 2]) / f12[ids]
        c = (3.0 * m - 2.0 * b[0:n - 1] - b[1:n]) / dx
        d = (b[0:n - 1] + b[1:n] - 2.0 * m) / dx ** 2
        c = np.concatenate([c, [0.0]]) 
        d = np.concatenate([d, [0.0]]) 
        self._coef = np.array([d, c, b, y])

class PChipSpline(CubicSpline):
    def _pchipendpoint(self, h1, h2, del1, del2):
        d = ((2 * h1 + h2) * del1 - h1 * del2) / (h1 + h2)
        if np.sign(d) != np.sign(del1):
            d = 0
        elif (np.sign(del1) != np.sign(del2)) and (_smooth_abs(d) > _smooth_abs(3 * del1)):
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
