import mpmath as mp
import numpy as np
import ad
import itertools as it

import etjj

def test_autodiff():
    Ninv = [ad_iv(x) for x in [0.01, 0.05, 0.2, 1.0]]
    ts = [0., 0.1, 0.5, 0.6, np.inf]
    sps = np.cumsum([ad_iv(0.5)**2 for _ in range(10)]).tolist()
    N = 10
    etjj_below = etjj.etjj_below(N, Ninv, ts, sps)
    for x in etjj_below.flat:
        for v in Ninv + sps:
            x.d(v)
            x.d2(v)
            print(v)

if __name__ == "__main__":
     Ninv = ad.adnumber(np.array([0.1, 0.2, 0.001, 0.003]))
     sp = ad.adnumber([0.01, 0.02, 0.03, 0.4, 0.6, 0.5, 0.6, 0.8, 1.01])
     x = etjj.etjj_above(10, Ninv, np.array([0., 0.1, 0.2, 1.0, np.inf]), sp)
