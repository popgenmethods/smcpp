import numpy as np
import json

class SMCModel(object):
    def __init__(self):
        self._x = self._hidden_states = self.N0 = self.theta = self.rho = None

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, _x):
        self._x = np.array(_x)

    @property
    def hidden_states(self):
        return self._hidden_states

    @hidden_states.setter
    def hidden_states(self, _hs):
        self._hidden_states = np.array(_hs)

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

    @property
    def M(self):
        return self._hidden_states.shape[0] - 1

    def to_json(self):
        return json.dumps({
            'x': [list(self.a), list(self.b), list(self.s)],
            'hidden_states': list(self.hidden_states),
            'N0': self.N0,
            'theta': self.theta,
            'rho': self.rho
            }, indent=4, sort_keys=True)

    @classmethod
    def from_file(klass, fn):
        obj = json.load(open(fn, "rt"))
        ret = klass()
        for attr in "x hidden_states N0 theta rho".split():
            setattr(ret, attr, obj[attr])
        return ret
