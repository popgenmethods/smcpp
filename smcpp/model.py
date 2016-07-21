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

    def reset_derivatives(self):
        self._y = self._y.astype('float')
        self._refit()

    @property
    def knots(self):
        return self._knots

    def to_d(self):
        return {'s': list(self._s), 'knots': list(self._knots),
                'y': list(self._y.astype('float')),
                'spline_class': self._spline_class.__name__}

    def __setitem__(self, key, item):
        self._y[key] = item[:]
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

    @property
    def split_ind(self):
        'Return k such that model2.t[k] <= split < model2.t[k + 1]'
        cs = np.cumsum(self._models[1]._knots)
        return np.searchsorted(cs, self._split)

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

    @property 
    def dlist(self):
        return self._models[0].dlist + self._models[1].dlist

    def to_d(self):
        return {'model1': self._models[0].to_d(),
                'model2': self._models[1].to_d(),
                'split': self._split}

    # FIXME this counts the part before the split twice
    def regularizer(self):
        return sum([m.regularizer() for m in self._models])

    def reset_derivatives(self):
        for m in self._models:
            m.reset_derivatives()

    def __getitem__(self, coords):
        a, cc = coords
        return self._models[a][cc]

    def __setitem__(self, coords, x):
        a, cc = coords
        self._models[a][cc] = x
        self.update_observers('model update')
