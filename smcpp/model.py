from __future__ import absolute_import, division, print_function
import numpy as np
import json
import logging
from ad import adnumber, ADV
logger = logging.getLogger(__name__)

from . import _smcpp, estimation_tools
from .spline import CubicSpline


class SMCModel(object):

    def __init__(self, s, knots):
        self._s = s
        self._knots = knots
        self.y = np.ones_like(knots)

    @property
    def s(self):
        return self._s

    @property
    def K(self):
        return len(self.knots)

    @property
    def knots(self):
        return self._knots

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, y):
        self._y = y
        self._spline = CubicSpline(self._knots, self._y)

    @property
    def dlist(self):
        return [yy for yy in self.y if isinstance(yy, ADV)]

    def regularizer(self):
        return self._spline.roughness()

    def stepwise_values(self):
        return self._spline.eval(np.cumsum(self._s))

    def to_dict(self):
        x = np.array([self.stepwise_values(), self.stepwise_values(), self._s])
        return {'x': x.astype('float').tolist(), 'y': self._y, 's': self._s, 'knots': self._knots}

    @classmethod
    def from_dict(klass, d):
        r = klass(d['s'], d['knots'])
        r.y = d['y']
        return r

    def copy(self):
        return SMCModel.from_dict(self.to_dict())
