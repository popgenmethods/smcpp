from __future__ import absolute_import, division, print_function
import numpy as np
import json
import logging
from ad import adnumber, ADF
import ad.admath

from . import _smcpp, estimation_tools, spline, logging
from .observe import Observable

logger = logging.getLogger(__name__)

class SMCModel(Observable):
    def __init__(self, s, knots, spline_class=spline.PChipSpline):
        Observable.__init__(self)
        self._spline_class = spline_class
        self._s = s
        self._knots = knots
        self.y = np.zeros_like(knots, dtype='object')
        self._refit()

    @property
    def s(self):
        return self._s

    @property
    def K(self):
        return len(self.knots)

    @property
    def knots(self):
        return self._knots

    def __setitem__(self, key, item):
        self._y[key] = item
        self._refit()
        self.update_observers('model update')
          
    def __getitem__(self, ind):
        return self._y[ind]

    def _refit(self):
        self._spline = self._spline_class(np.log(self._knots), self._y)

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, y):
        self._y = y
        self._refit()

    @property
    def dlist(self):
        return [yy for yy in self.y if isinstance(yy, ADF)]

    def regularizer(self):
        return self._spline.roughness()

    def stepwise_values(self):
        return np.array(ad.admath.exp(self._spline.eval(np.log(np.cumsum(self._s)))))

    def to_dict(self):
        x = np.array([self.stepwise_values(), self.stepwise_values(), self._s])
        return {'x': x.astype('float').tolist(), 'y': self._y, 's': self._s, 'knots': self._knots}

    @classmethod
    def from_dict(klass, d):
        r = klass(d['s'], d['knots'])
        r.y = d['y']
        return r

    @property
    def distinguished_model(self):
        return self

    def copy(self):
        return SMCModel.from_dict(self.to_dict())

class SMCTwoPopulationModel(Observable):
    def __init__(self, model1, model2, split):
        Observable.__init__(self)
        self._models = [model1, model2]
        self._split = split

    @property
    def split(self):
        return self._split

    @split.setter
    def split(self, x):
        self._split = x
        self.update_observers('model update')

    @property
    def model1(self):
        return self._models[0]

    @property
    def model2(self):
        return self._models[1]

    @property
    def distinguished_model(self):
        return self.model1

    def regularizer(self):
        return sum([m.regularizer() for m in self._models])

    def __setitem__(self, coords, x):
        for (a, cc), xx in zip(coords, x):
            self.models[a][cc] = xx
        self.update_observers('model update')
