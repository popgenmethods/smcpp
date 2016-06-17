from __future__ import absolute_import, division, print_function
import numpy as np
import json

class EstimationResult(object):
    def __init__(self):
        self.theta = self.rho = self._model = self.N0 = None

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        v = model.stepwise_values()
        self._model = np.array([v, v, model.s]).astype('float').tolist()

    def dump(self, filename):
        json.dump(self.__dict__, open(filename + ".json", "wt"))

    @classmethod
    def load(klass, fileobj):
        ret = klass()
        ret.__dict__.update(json.load(open(fileobj, "rt")))
        return ret
