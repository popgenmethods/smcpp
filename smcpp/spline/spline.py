import numpy as np


class Spline:
    'A spline of order p'
    def __init__(self, p, x, y):
        self._p = p
        self._x = x
        self._y = y
        self._coef = np.zeros([p + 1, len(x)], dtype=object)
        self._fit()

    def __setitem__(self, item, x):
        self._y[item] = x
        self._fit()

    def __getitem__(self, item):
        return self._y[item]

    def __call__(self, points):
        'Evaluate at points.'
        points = np.atleast_1d(points)
        ip = np.searchsorted(self._x, points, side="right") - 1
        ret = np.zeros(len(points), dtype=object)
        # The spline is constrained to be flat outside of the knot range
        ret[ip < 0] = self._coef[-1, 0]
        ret[ip >= len(self._x) - 1] = self._coef[-1, -1]
        good = (0 <= ip) & (ip < len(self._x) - 1)
        ipg = ip[good]
        p = np.arange(self._p + 1)[::-1, None]
        xi = (points[good] - self._x[ipg]) ** p
        ret[good] = (self._coef[:, ipg] * xi).sum(axis=0)
        return ret

    def _fit(self):
        raise NotImplementedError()

    def tv(self):
        return abs(np.diff(self._y)).sum()

    def roughness(self):
        return (np.diff(self._y, 2) ** 2).sum()
