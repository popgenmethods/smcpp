import numpy as np
from smcpp.moran_eigensystem import MoranEigensystem

def test_moran():
    N = 5
    def _check(me):
        assert np.linalg.norm(me._M - (me._U * me._D[None, :]).dot(me._Uinv)) < 1e-4
    for a in range(2):
        _check(MoranEigensystem(N, a))
    _check(MoranEigensystem(N))
