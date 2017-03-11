import scipy.optimize
import numpy as np


def box_constrain(x, bounds):
    return np.maximum(np.minimum(x, bounds[:, 1]), bounds[:, 0])


def AdaMax(f, x0, args, jac, alpha=0.1, b1=0.9, b2=0.999, eps=1e-3, **kwargs):
    assert jac

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
        theta += delta
        if kwargs.get('callback'):
            kwargs['callback'](theta)
    return scipy.optimize.OptimizeResult({'x': theta, 'fun': ft})


def Adam(f, x0, args, jac, alpha=0.01, b1=0.9, b2=0.999, eps=1e-8, **kwargs):
    def _f(x0):
        return tuple(q(x0, *args) for q in (f, jac))
    obj, grad = _f(x0)
    theta = x0.copy()
    t = 0
    mt = 0
    vt = 0
    while True:
        t += 1
        ft, gt = _f(theta)
        mt = b1 * mt + (1. - b1) * gt
        vt = b2 * vt + (1. - b2) * gt ** 2
        mt_hat = mt / (1. - b1 ** t)
        vt_hat = vt / (1. - b2 ** t)
        delta = -alpha * mt_hat / (vt ** .5 + eps)
        theta1 = theta + delta
        if max(abs(theta1 - theta)) < kwargs.get('options', {}).get('xtol', 1e-1):
            break
        theta = theta1
        if kwargs.get('callback'):
            kwargs['callback'](theta)
    return scipy.optimize.OptimizeResult({'x': theta, 'fun': ft})
