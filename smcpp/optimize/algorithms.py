def box_constrain(x, bounds):
    return np.maximum(np.minimum(x, bounds[:, 1]), bounds[:, 0])

def AdaMax(f, x0, args, jac, bounds, alpha=0.1, b1=0.9, b2=0.999, eps=1e-3, **kwargs):
    assert jac
    bounds = np.array(bounds)
    def _f(x0):
        return tuple(q(x0, *args) for q in (f, jac))
    obj, grad = _f(x0)
    theta = x0.copy()
    t = 0
    mt = 0
    ut = 0
    while True:
        t += 1
        ft, gt = _f(theta)
        mt = b1 * mt + (1. - b1) * gt
        ut = np.maximum(b2 * ut, abs(gt))
        delta = -(alpha / (1. - b1 ** t)) * mt / ut
        if np.linalg.norm(delta) < eps:
            break
        theta = box_constrain(theta + delta, bounds)
        if kwargs.get('callback'):
            kwargs['callback'](theta)
    return scipy.optimize.OptimizeResult({'x': theta, 'fun': ft})
