from __future__ import absolute_import, division, print_function
import numpy as np
import json
import logging
from ad import adnumber
logger = logging.getLogger(__name__)

from . import _smcpp, estimation_tools

class SMCModel(object):
    def __init__(self, s, exponential_pieces):
        self._exponential_pieces = exponential_pieces
        self._x = np.ones([3, len(s)], dtype=object)
        self._x[2] = s
        self._x[1] += 0.1
        self._flat_pieces = [i for i in range(self.K) if i not in exponential_pieces]
        self._add_derivatives()
        self._coords = [(a, b) for b in range(self.K) for a in ([0, 1] if b in self._exponential_pieces else [0])]
        pc = 1. / self.s
        pc[-1] = 1. / min(10, _smcpp.T_MAX - np.sum(self.s))
        self._precond = {c: float(pc[c[1]]) for c in self._coords}

    @property
    def coords(self):
        return self._coords

    @property
    def precond(self):
        return self._precond

    def regularizer(self, penalty):
        return estimation_tools.regularizer(self, penalty)

    def _add_derivatives(self):
        for i in range(2):
            for j in range(self.K):
                c = (i, j)
                self._x[c] = adnumber(float(self._x[c]), c)
        self._x[1, self.flat_pieces] = self._x[0, self.flat_pieces]

    def __getitem__(self, c):
        return self._x[c]

    def __setitem__(self, c, y):
        self._x[c] = y
        if c[1] in self._flat_pieces:
            self._x[1, c[1]] = y

    @property
    def flat_pieces(self):
        return self._flat_pieces
    
    @property
    def s(self):
        return self._x[2]

    @property
    def x(self):
        return self._x

    @property
    def K(self):
        return self.x.shape[1]

    def to_dict(self):
        return {'x': list(map(list, self.x)), 'exponential_pieces': self._exponential_pieces}

    @classmethod
    def from_dict(klass, d):
        r = klass(d['x'][2], d['exponential_pieces'])
        r.x[:2] = d['x'][:2]
        return r

    def copy(self):
        return SMCModel.from_dict(self.to_dict())

