import numpy as np


@np.vectorize
def smooth_abs(x):
    return (x**2 + 1e-3)**0.5


def polyval(c, x):
    ret = 0.
    for i in range(len(c)):
        ret = ret * x + c[i]
    return ret


