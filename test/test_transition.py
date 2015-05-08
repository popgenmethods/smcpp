import pytest
import numpy as np

@pytest.fixture
def pexp():
    K = 3
    ts = np.cumsum(ad.adnumber(np.array([0.] + sorted(np.random.exponential(2., size=K - 1)))))
    pc = np.zeros((1, K))
    qc = np.ones((1, K))
    b = ad.adnumber(np.random.normal(size=K))
    return PExp(ts, pc, qc, b)

def test_derivative_t():
    K = 3
    ts = ad.adnumber(np.array([0.] + sorted(np.random.exponential(2., size=K - 1))))
    pc = np.zeros((1, K))
    qc = np.ones((1, K))
    b = ad.adnumber(np.random.normal(size=K))
    pexp = PExp(np.cumsum(ts), pc, qc, b)
    R = pexp.integral0()
    d = np.array([0.0, 1.0, 2.0, 3.0, np.inf])
    rho = 1e-6
    tm = TransitionMatrix(rho, R, d)
    for i in range(tm.A.shape[0]):
        for j in range(tm.A.shape[0]):
            for k in range(1, pexp.ts.shape[0] - 1):
                db = tm.A[i, j].d(ts[k])
                eps = 2.0
                z = np.eye(ts.shape[0])[k] * eps
                # pexp = PExp(PPoly(pexp.p.c, tsp), PPoly(pexp.q.c, tsp), pexp.a, pexp.b)
                R2 = PExp(np.cumsum(ts + z), pc, qc, b).integral0()
                tm2 = TransitionMatrix(rho, R2, d)
                print(i, j, k, db, tm2.A[i, j], tm.A[i, j] + eps * db)
                diff = tm2.A[i, j] - (tm.A[i, j] + eps * db)
                if db == 0.0:
                    assert tm2.A[i, j] == tm.A[i, j]
                else:
                    assert -1e-3 < diff / eps < 1e-3

def test_derivative_b(pexp):
    R = pexp.integral0()
    d = np.array([0.0, 1.0, 2.0, 3.0, np.inf])
    rho = 1e-6
    tm = TransitionMatrix(rho, R, d)
    for i in range(tm.A.shape[0]):
        for j in range(tm.A.shape[0]):
            for k in range(pexp.b.shape[0]):
                db = tm.A[i, j].d(pexp.b[k])
                eps = 1.0
                z = np.eye(pexp.b.shape[0])[k] * eps
                # pexp = PExp(PPoly(pexp.p.c, tsp), PPoly(pexp.q.c, tsp), pexp.a, pexp.b)
                R2 = PExp(pexp.ts, pexp.p.c, pexp.q.c, pexp.b + z).integral0()
                tm2 = TransitionMatrix(rho, R2, d)
                print(i, j, k, db, tm2.A[i, j], tm.A[i, j] + eps * db)
                diff = tm2.A[i, j] - (tm.A[i, j] + eps * db)
                if db == 0.0:
                    assert tm2.A[i, j] == tm.A[i, j]
                else:
                    assert -1e-3 < diff / eps < 1e-3

def test_jacobian(pexp):
    R = pexp.integral0()
    d = np.array([0.0, 1.0, 2.0, 3.0, np.inf])
    rho = 1e-4
    tm = TransitionMatrix(rho, R, d)
    tm.jacobian(R.a)

def test_sum_to_one(pexp):
    R = pexp.integral0()
    d = np.array([0.0, 1.0, 2.0, 3.0, np.inf])
    tm = TransitionMatrix(1e-6, R, d)
    assert np.all(np.abs(tm.A.sum(axis=1) - 1) < 1e-6)
