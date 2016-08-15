import pytest
import numpy as np
import ad

import smcpp._smcpp, smcpp.model
from fixtures import im

def test_d(im):
    eps = 1e-8
    model = im.model
    K = model.K
    model[:] = [ad.adnumber(1. * k / K, tag=k) for k in range(1, K + 1)]
    print(model.stepwise_values())
    im.model = model
    im.E_step()
    im.Q()
    trans1 = im.transition
    print(trans1)
    M = trans1.shape[0]
    m2 = smcpp.model.SMCModel.from_dict(model.to_dict())
    im.model = m2
    for k in range(K):
        m2[k] += 1e-8
        im.model = m2
        im.Q()
        m2[k] -= 1e-8
        trans2 = im.transition
        for i in range(M):
            for j in range(M):
                try:
                    dx = trans1[i, j].d(model[k])
                except AttributeError:
                    dx = 0.
                dx2 = (trans2[i,j] - float(trans1[i,j])) / eps
                print(k, i, j, dx, dx2)
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
