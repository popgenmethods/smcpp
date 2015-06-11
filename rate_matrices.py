from __future__ import division
import numpy as np
import shelve
import logging

logger = logging.getLogger(__name__)

from _expm import expm

class MatrixInterpolator:
    def __init__(self, n, shelf_name, description):
	self._n = n
        self._interp_memo = shelve.open(shelf_name)
        self._desc = description

    def interpolators(self):
        key = str(self._n)
        if key not in self._interp_memo:
            logger.info("Interpolating %s matrix exponentials (only needs to happen once)" % self._desc)
            Q = self.rate_matrix()
            t = 1e-8
            # for a=2, absorbing state is N
            A = self._limit
            M = expm(Q * t)
            eps = 1.0
            np.set_printoptions(linewidth=100)
            u = []
            x = []
            while eps > 1e-8:
                logger.info("Interpolation progress: %.2f%%" % (1e-6 / eps));
                u.append(t)
                x.append(M)
                t += 0.05
                M = expm(Q * t)
                eps = np.linalg.norm(M - A)
            logger.info("Interpolation progress: Completed")
            u = np.array([0] + u + [np.inf])
            x = np.array([np.eye(Q.shape[0])] + x + [A])
            self._interp_memo[key] = (x, u)
        return self._interp_memo[key]

class MoranMatrixInterpolator(MatrixInterpolator):
    def __init__(self, n):
        MatrixInterpolator.__init__(self, n, ".moran.dat", "Moran") 
        _limit = np.zeros((n + 1, n + 1))
        _limit[:, -1] = 1.0

    def rate_matrix(self):
        M = np.zeros([self._n + 1, self._n + 1])
        for i in range(0, self._n + 1):
            if i > 0:
		M[i, i - 1] = 0.5 * k * (self._n - k)
		M[i, i] -= -M[i, i - 1]
            if i < self._n:
		M[i, i + 1] = 2 * (self._n - k) + 0.5 * k * (self._n - k)
		M[i, i] -= M[i, i + 1]
        return M

class AncestralProcessMatrixInterpolator(MatrixInterpolator):
    def __init__(self, n):
        MatrixInterpolator.__init__(self, n, ".ap.dat", "ancestral process") 
	self._limit = np.zeros([self._n - 1, self._n - 1])
	self._limit[:, 0] = 1.0

    def rate_matrix(self):
	M = np.zeros([self._n - 1, self._n - 1])
	for i in range(self._n - 1):
	    M[i, i - 1] = (i + 2) * (i + 1) / 2 - 1
	    M[i, i] = -M[i, i - 1]
        return M
