import numpy as np

def _moran_rate_matrix(N):
    ret = np.zeros([N + 1, N + 1])
    i, j = np.indices(ret.shape)
    # Superdiagonal
    k = np.arange(N)
    ret[j == i + 1] = 0.5 * k * (N - k)
    # Subdiagonal
    k += 1
    ret[j == i - 1] = 0.5 * k * (N - k)
    ret[i == j] = -np.sum(ret, axis=1)
    return ret

def _modified_rate_matrix(N, a):
    ret = np.zeros([N + 1, N + 1])
    i, j = np.indices(ret.shape)
    # Superdiagonal
    k = np.arange(N)
    ret[j == i + 1] = a * (N - k) + 0.5 * k * (N - k)
    # Subdiagonal
    k += 1
    ret[j == i - 1] = (2 - a) * k + 0.5 * k * (N - k)
    ret[i == j] = -np.sum(ret, axis=1)
    return ret

class MoranEigensystem:
    def __init__(self, N, a=None):
        if a is None:
            self._M = _moran_rate_matrix(N)
        else:
            self._M = _modified_rate_matrix(N, a)
        self._D, self._U = np.linalg.eig(self._M)
        self._Uinv = np.linalg.inv(self._U)

    def expm(self, R):
        return (self._U * (self._D * R)[None, :]).dot(self._Uinv)
