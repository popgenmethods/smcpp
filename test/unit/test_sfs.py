import pytest
import numpy as np
import ad

import smcpp._smcpp, smcpp.model
from fixtures import im

def test_d(im):
    eps = 1e-4
    model = im.model
    K = model.K
    model[:] = ad.adnumber(np.arange(1, K + 1, dtype=float) / K)
    y = model[:].copy()
    print(y)
    print(model.stepwise_values())
    im.model = model
    im.E_step()
    em1 = im.emission
    print(em1)
    M = em1.shape[0]
    I = np.eye(K)
    for k in range(K):
        aa = [float(_) for _ in y]
        aa += eps * I[k]
        model[:] = aa
        im.model = model
        im.Q()
        em2 = im.emission
        for i in range(M):
            for j in range(M):
                dx = em1[i, j].d(y[k])
                dx2 = (em2[i,j] - float(em1[i,j])) / eps
                print(k, i, j, dx, dx2, (dx - dx2) / dx)
    assert False
