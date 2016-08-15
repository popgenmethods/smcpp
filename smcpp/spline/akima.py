from __future__ import absolute_import, division, print_function
import numpy as np

from .common import smooth_abs
from .cubic import CubicSpline

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
        dm = smooth_abs(np.diff(m1))
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
