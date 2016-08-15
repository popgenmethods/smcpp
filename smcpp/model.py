from __future__ import absolute_import, division, print_function
import numpy as np
import ad.admath

from . import spline, logging
from .observe import Observable


logger = logging.getLogger(__name__)


# Dummy class used for JCSFS and a few other places
class PiecewiseModel(object):
    def __init__(self, s, a):
        self.s = s
        self.a = a

    def stepwise_values(self):
        return self.a

    def __getitem__(self, it):
        return self.a[it]

    def __setitem__(self, it, x):
        self.a[it] = x

    @property
    def dlist(self):
        ret = []
        for yy in self.a:
            try:
                ret += [d for d in yy.d() if d.tag is not None]
            except AttributeError:
                pass
        return ret


class SMCModel(Observable):
    def __init__(self, s, knots, spline_class=spline.PChipSpline):
        Observable.__init__(self)
        self._spline_class = spline_class
        self._s = np.array(s)
        self._cumsum_s = np.cumsum(s)
        self._knots = np.array(knots)
        # self._trans = np.vectorize(lambda x: x)
        self._trans = np.log
        self._spline = self._spline_class(self._trans(self._knots))

    @property
    def s(self):
        return self._s

    @property
    def K(self):
        return len(self.knots)

    def reset_derivatives(self):
        self[:] = self[:].astype('float').astype('object')

    def refit(self):
        y = self[:]
        self._spline = self._spline_class(self._trans(self._knots))
        self[:] = y

    @property
    def knots(self):
        return self._knots

    def __setitem__(self, key, item):
        self._spline[key] = item
        self.update_observers('model update')

    def __getitem__(self, key):
        return self._spline[key]

    @property
    def dlist(self):
        ret = []
        for yy in self[:]:
            try:
                ret += [d for d in yy.d() if d.tag is not None]
            except AttributeError:
                pass
        return ret

    def regularizer(self):
        # ret = self._spline.integrated_curvature()
        ret = self._spline.tv()
        if not isinstance(ret, ad.ADF):
            ret = ad.adnumber(ret)
        return ret

    def __call__(self, x):
        'Evaluate :self: at points x.'
        return np.array(
            ad.admath.exp(self._spline(self._trans(x)))
        )

    def stepwise_values(self):
        return self(np.cumsum(self._s))

    def reset(self):
        self[:] = 0.

    def to_s(self, until=None):
        ret = []
        for ary in [self[:until], self.stepwise_values()]:
            ary = ary.astype('float')
            fmt = " ".join(["{:>5.2f}"] * len(ary))
            ret.append(fmt.format(*ary))
        return "\n" + "\n".join(ret)

    def to_dict(self):
        return {
                'class': self.__class__.__name__,
                's': list(self._s),
                'knots': list(self._knots),
                'spline_class': self._spline_class.__name__,
                'y': self[:].astype('float').tolist()
                }

    @classmethod
    def from_dict(cls, d):
        assert cls.__name__ == d['class']
        spc = getattr(spline, d['spline_class'])
        r = cls(d['s'], d['knots'], spc)
        r[:] = d['y']
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
    def split_ind(self):
        'Return k such that model2.t[k] <= split < model2.t[k + 1]'
        cs = np.cumsum(self._models[1]._knots)
        return np.searchsorted(cs, self._split) + 1

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

    def reset(self):
        for m in self._models:
            m.reset()

    def to_dict(self):
        return {'class': self.__class__.__name__,
                'model1': self._models[0].to_dict(),
                'model2': self._models[1].to_dict(),
                'split': float(self._split)}

    @classmethod
    def from_dict(cls, d):
        assert cls.__name__ == d['class']
        model1 = SMCModel.from_dict(d['model1'])
        model2 = SMCModel.from_dict(d['model2'])
        return cls(model1, model2, d['split'])

    def to_s(self):
        return "\nPop. 1:\n{}\nPop. 2:\n{}\nSplit: {:.3f}".format(
            self._models[0].to_s(), self._models[1].to_s(self.split_ind),
            self.split)

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
