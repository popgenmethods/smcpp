import numpy as np


@np.vectorize
def smooth_abs(x):
    return (x**2 + 1e-3)**0.5




