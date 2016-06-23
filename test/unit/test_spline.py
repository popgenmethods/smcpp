import ad
import numpy as np
import smcpp.spline

def test_spline():
    for klass in (smcpp.spline.CubicSpline, smcpp.spline.AkimaSpline):
        _test_spline(klass)

def _test_spline(klass):
    x = np.arange(20) / 20.
    y = ad.adnumber(np.random.normal(size=20))
    s = (np.arange(1, 20) - 0.3) / 20.
    sp = klass(x, y)
    pts = sp.eval(s)
    for i in range(20):
        y2 = np.array(y).astype('float')
        y2[i] += 1e-8
        sp2 = klass(x, y2)
        pts2 = sp2.eval(s)
        for p1, p2 in zip(pts, pts2):
            g1 = p1.d(y[i])
            g2 = (p2 - p1.x) * 1e8
            assert abs(g2 - g1) < 1e-5
