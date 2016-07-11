import pytest
import numpy as np
import ad

import smcpp._smcpp, smcpp.model

def test_rate_function():
    K = 10
    s = np.diff(np.logspace(np.log10(.01), np.log10(3.), 41))
    model = smcpp.model.SMCModel(s, np.logspace(np.log10(.01), np.log10(3.), K))
    model[:] = [ad.adnumber(0.002676322760403453),ad.adnumber(-0.010519987448975402)
            ,ad.adnumber(0.006727140517177145),ad.adnumber(0.0031333684894676865)
            ,ad.adnumber(-0.02302979056648467),ad.adnumber(0.0026368097606793172)
            ,ad.adnumber(0.0019921562626012993),ad.adnumber(0.004958301100037235)
            ,ad.adnumber(0.003199704865436452),ad.adnumber(0.0050129872575249744)]
    hidden_states = np.concatenate([[0.], np.logspace(-2, 1, 10), [np.inf]])
    eta = smcpp._smcpp.PyRateFunction(model, hidden_states)
    t = 1.08
    Rt = eta.R(t)
    for k in range(K):
        dq = Rt.d(model[k])
        model[k] += 1e-8
        Rt1 = smcpp._smcpp.PyRateFunction(model, hidden_states).R(t)
        a = float(Rt1 - Rt) * 1e8
        print(k, a, dq)
        model[k] -= 1e-8
