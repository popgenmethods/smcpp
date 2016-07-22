import pytest
import numpy as np
import ad

from fixtures import *

import smcpp._smcpp, smcpp.model

def test_d(im):
    eps = 1e-8
    model = im.model
    K = len(model[:])
    model[:] = ad.adnumber(np.random.normal(0, .1, size=K))
    y = model[:].copy()
    im.model = model
    im.E_step()
    im.Q()
    e1 = im.emission
    I = np.eye(K)
    for k in range(K):
        aa = [float(_) for _ in y]
        aa += eps * I[k]
        model[:] = aa
        im.model = model
        im.Q()
        e2 = im.emission
        for i in range(e1.shape[0]):
            for j in range(e1.shape[1]):
                d1 = e1[i, j].d(y[k])
                d2 = (e2[i,j] - float(e1[i,j])) / eps
                # print(k, i, j, (d1 - d2) / d2 if d2 != 0.0 else d1 == d2)

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
