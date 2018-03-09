import numpy as np
from .spline import Spline


class Piecewise(Spline):
    def __init__(self, x, y=None):
        if y is None:
            y = np.zeros_like(x, dtype='object')
        Spline.__init__(self, 0, x, y)

    def _fit(self):
        self._coef[:] = self._y
