import numpy as np
import json
import logging
logger = logging.getLogger(__name__)

from . import _smcpp, estimation_tools

class SMCModel(object):
    def __init__(self, s, exponential_pieces):
        self._exponential_pieces = exponential_pieces
        self._x = np.ones([3, len(s)])
        self.s = s
        self._flat_pieces = [i for i in range(self.K) if i not in exponential_pieces]
        self.b = self.a + 0.1
        self.flatten()

    @property
    def coords(self):
        return [(aa, j) for j in range(self.K) for aa in ((0,) if j in self.flat_pieces else (0, 1))]

    @property
    def precond(self):
        ret = {coord: 1. / self.s[coord[1]] for coord in self.coords}
        if (self.K - 1, 0) in self.coords:
            ret[(self.K - 1, 0)] = 1. / (_smcpp.T_MAX - np.sum(self.s))
        return ret

    def regularizer(self, penalty):
        return estimation_tools.regularizer(self, penalty)

    def flatten(self):
        self.b[self.flat_pieces] = self.a[self.flat_pieces]

    @property
    def flat_pieces(self):
        return self._flat_pieces

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, _x):
        self._x = np.array(_x)

    @property
    def a(self):
        return self.x[0]

    @a.setter
    def a(self, _a):
        self.x[0] = _a

    @property
    def b(self):
        return self.x[1]

    @b.setter
    def b(self, _b):
        self.x[1] = _b

    @property
    def s(self):
        return self.x[2]

    @s.setter
    def s(self, _s):
        self.x[2] = _s

    @property
    def K(self):
        return self.x.shape[1]

    def to_dict(self):
        return {'x': list(map(list, self.x)), 'exponential_pieces': self._exponential_pieces}
