import ad
import os
import numpy as np
import scipy.optimize
from abc import abstractmethod

from smcpp.observe import Observable
from smcpp.logging import getLogger
from smcpp.optimize.plugins.optimizer_plugin import OptimizerPlugin
import smcpp.optimize.algorithms
from .exceptions import *

logger = getLogger(__name__)


def _sigmoid(x, bounds):
    b, B = np.array(bounds).T
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


class AbstractOptimizer(Observable):
    '''
    Abstract representation of the execution flow of the optimizer.
    '''
    def __init__(self, analysis, algorithm, xtol, ftol, blocks, solver_args={}):
        Observable.__init__(self)
        self._analysis = analysis
        self._algorithm = algorithm
        self._ftol = ftol
        self._xtol = xtol
        self._blocks = blocks
        self._solver_args = solver_args

    @abstractmethod
    def _coordinates(self, i):
        'Return a list of groups of coordinates to be optimized at iteration i.'
        return []

    @abstractmethod
    def _bounds(self, coords):
        'Return a list of bounds for each coordinate in :coords:.'
        return []

    # In the one population case, this method adds derivative information to x
    def _prepare_x(self, x):
        return [ad.adnumber(xx, tag=i) for i, xx in enumerate(x)]

    def _f(self, x, analysis, coords, bounds, k=None):
        x = self._prepare_x(x)
        xs = _sigmoid(x, bounds)
        logger.debug("x: " + ", ".join(["%.1f" % float(xx) for xx in xs]))
        analysis.model[coords] = xs
        q = analysis.Q(k)
        # autodiff doesn't like multiplying and dividing inf
        if np.isinf(q.x):
            return [np.inf, np.zeros(len(x))]
        q = -q
        ret = [q.x, np.array(list(map(q.d, x)))]
        return ret

    def _minimize(self, x0, coords, bounds):
        self._xk = None
        if os.environ.get("SMCPP_GRADIENT_CHECK"):
            print("\n\ngradient check")
            y, dy = self._f(x0, self._analysis, coords)
            for i in range(len(x0)):
                x0[i] += 1e-8
                y1, _ = self._f(x0, self._analysis, coords)
                print("***grad", i, y1, (y1 - y) * 1e8, dy[i])
                x0[i] -= 1e-8
        try:
            try:  # Adam/AdaMax
                alg = getattr(smcpp.optimize.algorithms, self._algorithm)
            except AttributeError:
                alg = self._algorithm
            options = {
                    'xtol': self._xtol, 'ftol': self._ftol, 'factr': 1e1, 'gtol': 10.
                    }
            res = scipy.optimize.minimize(self._f, np.zeros_like(x0),
                    jac=True,
                    args=(self._analysis, coords, bounds),
                    options=options,
                    method=alg)
            res.x = _sigmoid(res.x, bounds)
            return res
        except ConvergedException:
            logger.debug("Converged: |xk - xk_1| < %g", self._xtol)
            return scipy.optimize.OptimizeResult(
                {'x': self._xk, 'fun': self._f(self._xk, self._analysis, coords)[0]})

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
                    x0 = self._analysis.model[coords]
                    bounds = np.transpose([x0 - 2., x0 + 2.])
                    logger.debug("bounds: %s", bounds)
                    # bounds = self._bounds(coords)
                    res = self._minimize(x0, coords, bounds)
                    self.update_observers('post minimize',
                                          coords=coords,
                                          res=res, **kwargs)
                    if os.environ.get("SMCPP_DEBUG"):
                        y = input("Break? ")
                        if y[0] == "y":
                            im = list(self._analysis._ims.values())[0]
                            gs = im.gamma_sums
                            z = np.zeros_like(list(gs[0].values())[0])
                            d = {}
                            for g in gs:
                                for k in g:
                                    d.setdefault(k, z.copy())
                                    d[k] += g[k]
                            gs = d
                            xis = np.sum(im.xisums, axis=0)
                            import ipdb
                            ipdb.set_trace()
                    self._analysis.model[coords] = res.x
                    self.update_observers('post mini M-step',
                                          coords=coords,
                                          res=res, **kwargs)
                self.update_observers('post M-step', **kwargs)
        except EMTerminationException:
            pass
        # Conclude the optimization and perform any necessary callbacks.
        self.update_observers('optimization finished')

    def _callback(self, xk):
        if self._xk is None:
            self._xk = xk
            return
        delta = max(abs(xk - self._xk))
        self._xk = xk
        if delta < self._xtol:
            raise ConvergedException()

    def update_observers(self, *args, **kwargs):
        kwargs.update({
            'optimizer': self,
            'analysis': self._analysis,
            'model': self._analysis.model})
        Observable.update_observers(self, *args, **kwargs)


class SMCPPOptimizer(AbstractOptimizer):
    'Model fitting for one population.'

    def __init__(self, analysis, algorithm, xtol, ftol, blocks, solver_args):
        AbstractOptimizer.__init__(self, analysis, algorithm, xtol, ftol, blocks, solver_args)
        for cls in OptimizerPlugin.__subclasses__():
            try:
                if not cls.DISABLED:
                    self.register(cls())
            except TypeError:
                # Only register listeners with null constructor
                pass

    def _coordinates(self):
        model = self._analysis.model
        ret = []
        K = model.K
        if self._blocks is None:
            self._blocks = min(4, K)
        if not 1 <= self._blocks <= K:
            logger.error("blocks must be between 1 and K")
            sys.exit(1)
        r = list(range(K))
        return [r]
        ret = [r[a:a+self._blocks] for a in range(K - self._blocks + 1)]
        if r not in ret:
            ret.append(r)
        ret = ret[::-1]
        logger.debug("block schedule: %s", str(ret))
        return ret

    def _bounds(self, coords):
        ret = np.log([self._analysis._bounds] * len(coords))
        return ret


class TwoPopulationOptimizer(SMCPPOptimizer):
    'Model fitting for two populations.'

    def _coordinates(self):
        coords = super()._coordinates()
        coords2 = []
        si = self._analysis.model.split_ind
        for c in coords:
            c = np.array(c)
            c = c[c < si]
            if c.size:
                coords2.append(c)
        return [(0, coords), (1, coords2)]

    def _bounds(self, coords):
        return SMCPPOptimizer._bounds(self, coords[1])
