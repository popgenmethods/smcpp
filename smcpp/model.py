from textwrap import dedent
import numpy as np
import ad.admath
import os
import wrapt
import scipy.optimize

from . import spline, logging, util
from .observe import Observable, Observer, targets
import smcpp.defaults

logger = logging.getLogger(__name__)

def tag_sort(s):
    return sorted(set(s), key=lambda x: x.tag)

@wrapt.decorator
def returns_ad(wrapped, instance, args, kwargs):
    ret = wrapped(*args, **kwargs)
    if not isinstance(ret, ad.ADF):
        ret = ad.adnumber(ret)
    return ret

class BaseModel(Observable):
    def __init__(self, N0, pid):
        Observable.__init__(self)
        self._N0 = N0
        self._pid = pid

    @property
    def N0(self):
        return self._N0

    @property
    def pid(self):
        return self._pid

    @returns_ad
    def regularizer(self):
        # curvature
        a = self.stepwise_values()
        y = ad.admath.log(a)
        rd = smcpp.defaults.regularization_degree
        cs = np.cumsum(self.s)[:-rd]
        d1 = np.diff(y, rd)
        r1 = abs(d1 ** rd).sum() ** (1. / rd)
        r2 = self._spline.roughness() ** .5
        logger.debug("r1:%f r2:%f", float(r1), float(r2))
        return r2


# Dummy class used for JCSFS and a few other places
class PiecewiseModel(BaseModel):
    def __init__(self, a, s, N0, pid=None):
        super().__init__(N0, pid)
        assert len(a) == len(s)
        self.s = np.array(s)
        self.a = np.array(a)

    @property
    def distinguished_model(self):
        return self
        
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
        return tag_sort(ret)


class OldStyleModel(PiecewiseModel):
    def __init__(self, a, b, s, N0):
        assert b[-1] == a[-1]
        ap = []
        sp = []
        for aa, bb, ss, cs in zip(a, b, s, util.cumsum0(s)[:-1]):
            if aa == bb:
                ap.append(aa)
                sp.append(ss)
            else:
                s0 = cs if cs > 0 else 1e-5
                s1 = s0 + ss
                # t = np.logspace(np.log10(s0), np.log10(s1), 40)
                t = np.linspace(s0, s1, 40)
                sp += np.diff(t).tolist()
                ap += (aa * (bb / aa) ** ((t[:-1] - s0) / (s1 - s0))).tolist()
        super().__init__(ap, sp, N0)


class SMCModel(BaseModel):
    def __init__(self, knots, N0, spline_class=spline.CubicSpline, pid=None):
        super().__init__(N0, pid)
        self._spline_class = spline_class
        self._knots = np.array(knots)
        self._s = np.r_[
                self._knots[:1],
                np.diff(
                    np.logspace(np.log10(self._knots[0]),
                                np.log10(self._knots[-1]),
                                smcpp.defaults.pieces)
                    )
                ]
        self._trans = np.log
        # self._trans = lambda x: x
        self._spline = self._spline_class(self.transformed_knots)

    def for_pop(self, pid):
        assert pid == self.pid
        return self

    @property
    def s(self):
        return self._s

    @property
    def K(self):
        return len(self.knots)

    def randomize(self):
        logger.debug("model before randomization: %s", self[:].astype('float'))
        self[:] += np.random.normal(0., .0001, size=len(self[:]))
        logger.debug("model after randomization: %s", self[:].astype('float'))

    @property
    def knots(self):
        return self._knots

    @property
    def transformed_knots(self):
        return self._trans(self._knots)

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
        return tag_sort(ret)

    def __call__(self, x):
        'Evaluate :self: at points x.'
        ret = np.array(
            ad.admath.exp(self._spline(self._trans(x)))
        )
        return ret

    def match(self, other_model):
        a = np.cumsum(self.s)
        def f(x):
            self[:] = x
            return ((self(a).astype('float') - other_model(a).astype('float')) ** 2).sum()
        res = scipy.optimize.minimize(f, self[:].astype('float'))
        # logger.debug(res)
        self[:] = res.x

    def stepwise_values(self):
        return self(np.cumsum(self.s))

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
        d = {}
        d.update({
            'class': self.__class__.__name__,
            'knots': list(self._knots),
            'N0': self.N0,
            'spline_class': self._spline_class.__name__,
            'y': self[:].astype('float').tolist(),
            'pid': self.pid
            })
        return d

    @classmethod
    def from_dict(cls, d):
        assert cls.__name__ == d['class']
        spc = getattr(spline, d['spline_class'])
        r = cls(d['knots'], d['N0'], spc, d['pid'])
        r[:] = d['y']
        return r

    @property
    def distinguished_model(self):
        return self

    def copy(self):
        return SMCModel.from_dict(self.to_dict())


class SMCTwoPopulationModel(Observable, Observer):

    def __init__(self, model1, model2, split):
        Observable.__init__(self)
        self._models = [model1, model2]
        model1.register(self)
        model2.register(self)
        self._split = split

    @property
    def distinguished_model(self):
        return self.model1

    def for_pop(self, pid):
        if pid == None:
            # Special value indicating distinguished model when both lineages are apart.
            s = self.model1.s
            a = self.model1.stepwise_values()
            cs = util.cumsum0(self.model1.s)
            cs[-1] = np.inf
            ip = np.searchsorted(cs, self._split)
            sp = np.diff(np.insert(cs, ip, self._split))
            sp[-1] = 1.
            s = sp[ip - 1:]
            s[0] = self.split
            a = np.insert(a[ip - 1:], 0, np.inf)
            return PiecewiseModel(a, s, None)
        i = self.pids.index(pid)
        if i == 0:
            return self.model1
        else:
            assert i == 1
            assert self.model1.N0 == self.model2.N0
            assert self.model1._spline_class is self.model2._spline_class
            k1, k2 = [np.searchsorted(m.knots, self.split) for m in (self.model1, self.model2)]
            kts = np.r_[self.model2.knots[:k2], [self.split], self.model1.knots[k1 + 1:]]
            m = SMCModel(kts, self.model1.N0, self.model2._spline_class, self.model2.pid)
            m[:k2] = self.model2[:k2]
            m[k2] = ad.admath.log(self.model1(self.split).item())
            m[k2 + 1:] = self.model1[k1 + 1:]
            return m
            # return _concat_models(self.model1, self.model2, self.split)

    # Propagate changes from submodels up
    @targets('model update')
    def update(self, message, *args, **kwargs):
        self.update_observers('model update')

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
        return np.searchsorted(
            self.model2.knots,
            self._split, side="right") - 1

    @property
    def s(self):
        return self.model1.s

    @property
    def model1(self):
        return self._models[0]

    @property
    def K(self):
        return self.model1.K

    @property
    def model2(self):
        return self._models[1]

    @property
    def pids(self):
        return [m.pid for m in self._models]

    @property
    def dlist(self):
        return tag_sort(self._models[0].dlist + self._models[1].dlist)

    def randomize(self):
        for m in self._models:
            m.randomize()

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
        return dedent("""
        Pop. 1: {}
        Pop. 2: {}
        Split: {:.3f}
        """).format(
            self._models[0].to_s(), self._models[1].to_s(self.split_ind),
            self.split)

    # FIXME this counts the part before the split twice
    @returns_ad
    def regularizer(self):
        return sum([self.for_pop(pid).regularizer() for pid in self.pids])

    def __getitem__(self, coords):
        if isinstance(coords, slice):
            if coords != slice(None, None, None):
                raise RuntimeException()
            return np.concatenate([self.model1[:], self.model2[:]])
        a, cc = coords
        return self._models[a][cc]

    def __setitem__(self, coords, x):
        if isinstance(coords, slice):
            if coords != slice(None, None, None):
                raise RuntimeException()
            l = len(self.model1[:])
            self.model1[:] = x[:l]
            self.model2[:] = x[l:]
            return
        a, cc = coords
        # This will generate 'model updated' messages in the submodels.
        self._models[a][cc] = x


def _concat_models(m1, m2, t):
    if m1.N0 != m2.N0:
        raise RuntimeException()
    cs1 = np.cumsum(m1.s)
    cs2 = np.cumsum(m2.s)
    sv1 = m1.stepwise_values()
    sv2 = m2.stepwise_values()
    s = np.concatenate([m2.s[cs2 <= t], m1.s[cs1 > t]])
    a = np.concatenate([sv2[cs2 <= t], sv1[cs1 > t]])
    return PiecewiseModel(a, s, m2.N0, m2.pid)
