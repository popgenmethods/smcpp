import ad
import os
import numpy as np
import scipy.optimize
import itertools
from abc import abstractmethod

import smcpp.defaults
from smcpp.observe import Observable
from smcpp.logging import getLogger
from smcpp.optimize.plugins.optimizer_plugin import OptimizerPlugin
import smcpp.optimize.algorithms
from .exceptions import *

logger = getLogger(__name__)


class AbstractOptimizer(Observable):
    '''
    Abstract representation of the execution flow of the optimizer.
    '''
    def __init__(self, analysis, algorithm, xtol, ftol):
        Observable.__init__(self)
        self._plugins = []
        self._analysis = analysis
        self._algorithm = algorithm
        self._ftol = ftol
        self._xtol = xtol

    @abstractmethod
    def _coordinates(self, i):
        'Return a list of groups of coordinates to be optimized at iteration i.'
        return []

    def __getitem__(self, coords):
        return self._analysis.model[coords]

    def __setitem__(self, coords, x):
        self._analysis.model[coords] = x

    # In the one population case, this method adds derivative information to x
    def _prepare_x(self, x):
        return [ad.adnumber(xx, tag=i) for i, xx in enumerate(x)]

    def _sigmoid(self, x):
        # x = [xx / ss for xx, ss in zip(x, self._scale)]
        b, B = np.array(self._bounds).T
        s = []
        for xx in x:
            if xx > 0:
                z = ad.admath.exp(-xx)
                s.append(1 / (1 + z))
            else:
                z = ad.admath.exp(xx)
                s.append(z / (1 + z))
        s = np.array(s)
        return (B - b) * s + b


    def _f(self, x, analysis, coords):
        x = self._prepare_x(x)  # do not change this line
        xs = self._sigmoid(x)
        logger.debug("x: " + ", ".join(["%.3f" % float(xx) for xx in xs]))
        self[coords] = xs
        q = analysis.Q()
        # autodiff doesn't like multiplying and dividing inf
        if np.isinf(q.x):
            return [np.inf, np.zeros(len(x))]
        q = -q
        ret = [q.x, np.array(list(map(q.d, x)))]
        self._f_dict[tuple(np.array(x).astype('float').tolist())] = q.x
        return ret

    def _minimize(self, x0, coords):
        self._xk = self._k = self._delta = None
        try:
            try:  # Adam/AdaMax
                alg = getattr(smcpp.optimize.algorithms, self._algorithm)
            except AttributeError:
                alg = self._algorithm
            options = {
                    'xtol': self._xtol, 'ftol': self._ftol, 'disp': True
                    }
            x0z = np.zeros_like(x0)
            # preconditioner
            # self._scale = 1.
            # f, dq = self._f(x0z, self._analysis, coords)
            # self._scale = (1 + np.abs(dq)) / 10.
            # logger.debug("scale: %s", self._scale.round(1))
            if os.environ.get("SMCPP_GRADIENT_CHECK"):
                # TODO move to plugin
                print("\n\ngradient check")
                y, dy = self._f(x0, self._analysis, coords)
                for i in range(len(x0)):
                    x0[i] += 1e-8
                    y1, _ = self._f(x0, self._analysis, coords)
                    print("***grad", i, y1, (y1 - y) * 1e8, dy[i])
                    x0[i] -= 1e-8
            y = self[coords]
            self._f_dict = {}
            self._last_f = None
            f0 = self._f(x0z, self._analysis, coords)[0]
            if len(y) > 1:
                res = scipy.optimize.minimize(self._f, x0z,
                        jac=True,
                        args=(self._analysis, coords),
                        options=options,
                        bounds=[[-3, 3]] * len(x0z),
                        # callback=self._callback,
                        method=alg)
            else:
                def _f_scalar(x, *args, **kwargs):
                    return self._f(np.array([x]), *args, **kwargs)[0]
                res = scipy.optimize.minimize_scalar(_f_scalar,
                        bounds=[-3, 3],
                        options={'xtol': self._xtol, 'ftol': self._ftol},
                        args=(self._analysis, coords),
                        method='bounded')
                res.x = np.array([res.x])
            improv = (f0 - res.fun) / abs(f0)
            logger.debug("%% improvement in f=%f", improv)
            # if improv < self._ftol:
            #     logger.debug("improvement < %f=ftol; keeping old x", self._ftol)
            #     res.x = x0z
            res.x = self._sigmoid(res.x)
            return res
        except ConvergedException as ce:
            logger.debug("Converged: %s", str(ce))
            return scipy.optimize.OptimizeResult(
                {'x': self._sigmoid(self._xk),
                 'fun': self._f_dict[self._xk]}
                )

    def run(self, niter):
        self.update_observers('begin')
        try:
            for i in range(niter):
                # Perform E-step
                kwargs = {'i': i, 'niter': niter}
                self.update_observers('pre E-step', **kwargs)
                self._analysis.E_step()
                self.update_observers('post E-step', **kwargs)
                # Perform M-step
                self.update_observers('pre M-step', **kwargs)
                coord_list = self._coordinates()
                for coords in coord_list:
                    self.update_observers('M step', coords=coords, **kwargs)
                    x0 = self[coords]
                    self._bounds = np.transpose(
                        [np.maximum(x0 - 2., np.log(smcpp.defaults.minimum)),
                         np.minimum(x0 + 2., np.log(smcpp.defaults.maximum))])
                    logger.debug("bounds: %s", self._bounds)
                    res = self._minimize(x0, coords)
                    self.update_observers('post minimize',
                                          coords=coords,
                                          res=res, **kwargs)
                    self[coords] = res.x
                    self.update_observers('post mini M-step',
                                          coords=coords,
                                          res=res, **kwargs)
                self.update_observers('post M-step', **kwargs)
        except EMTerminationException:
            pass
        # Conclude the optimization and perform any necessary callbacks.
        self.update_observers('optimization finished')

    def _callback(self, xk):
        return
        if self._k is None:
            self._k = 1
        if self._k > 10:
            raise ConvergedException("Max_iter > %d" % 10)
        if self._xk is None:
            self._xk = xk
            return
        xk0 = self._sigmoid(self._xk)
        self._xk = xk
        xk = self._sigmoid(xk)
        if self._delta is None:
            self._delta = max(abs(xk - xk0))
            return
        self._delta = .2 * self._delta + .8 * max(abs(xk - xk0))
        logger.debug("delta: %f", self._delta)
        if self._delta < self._xtol:
            raise ConvergedException("delta=%f < xtol=%f" % (self._delta, self._xtol))

    def register_plugin(self, p):
        self._plugins.append(p)
        self.register(p)

    def update_observers(self, *args, **kwargs):
        kwargs.update({
            'optimizer': self,
            'analysis': self._analysis,
            'model': self._analysis.model})
        Observable.update_observers(self, *args, **kwargs)


class SMCPPOptimizer(AbstractOptimizer):
    'Model fitting for one population.'

    def __init__(self, analysis, algorithm, xtol, ftol):
        AbstractOptimizer.__init__(self, analysis, algorithm, xtol, ftol)
        for cls in OptimizerPlugin.__subclasses__():
            try:
                if not cls.DISABLED:
                    self.register_plugin(cls())
            except TypeError:
                # Only register listeners with null constructor
                pass

    def _coordinates(self):
        model = self._analysis.model
        K = model.K - 1
        # return [[k] for k in range(K)][::-1]
        return [list(range(K))] # + [list(range(K // 3))]

class TwoPopulationOptimizer(SMCPPOptimizer):
    'Model fitting for two populations.'

    def _coordinates(self):
        coords = super()._coordinates()
        si = self._analysis.model.split_ind
        ret = []
        for c in super()._coordinates():
            ret.append([0, c])
            c = np.array(c)
            c = c[c < si]
            if c.size:
                ret.append([1, c])
        return ret
