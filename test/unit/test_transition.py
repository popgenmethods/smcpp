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
    im.Q()
    trans1 = im.transition
    print(trans1)
    M = trans1.shape[0]
    I = np.eye(K)
    for k in range(K):
        aa = [float(_) for _ in y]
        aa += eps * I[k]
        model[:] = aa
        im.model = model
        im.Q()
        trans2 = im.transition
        for i in range(M):
            for j in range(M):
                dx = trans1[i, j].d(y[k])
                dx2 = (trans2[i,j] - float(trans1[i,j])) / eps
                print(k, i, j, dx, dx2, (dx - dx2) / dx)
    assert False

# def test_equal_jac_nojac(constant_demo_1, hs):
#     from timeit import default_timer as timer
#     start = timer()
#     trans1, jac = _pypsmcpp.transition(constant_demo_1, hs, rho, True)
#     end = timer()
#     print("Run-time with jacobian: %f" % (end - start))
#     start = timer()
#     trans2 = _pypsmcpp.transition(constant_demo_1, hs, rho, False)
#     end = timer()
#     print("Run-time without: %f" % (end - start))
#     assert np.allclose(trans1 - trans2, 0)
# 
# def test_sum_to_one(constant_demo_1, hs):
#     trans1 = _pypsmcpp.transition(constant_demo_1, hs, rho, False)
#     assert np.allclose(np.sum(trans1, axis=1), 1.0)
