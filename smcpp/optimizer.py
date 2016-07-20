from __future__ import absolute_import, division, print_function
import numpy as np
import scipy.optimize
import os.path, os
import ad, ad.admath
from abc import ABCMeta, abstractmethod

from . import estimation_tools, logging
from .observe import Observer, Observable

logger = logging.getLogger(__name__)


class AbstractOptimizer(Observable):
    '''
    Abstract representation of the execution flow of the optimizer.
    '''
    def __init__(self, analysis):
        Observable.__init__(self)
        self._analysis = analysis

    @abstractmethod
    def _coordinates(self, i):
        'Return a list of groups of coordinates to be optimized at iteration i.'
        return []

    @abstractmethod
    def _bounds(self, coords):
        'Return a list of bounds for each coordinate in :coords:.'
        return []

    def _f(self, x, analysis, coords):
        x = ad.adnumber(x)
        analysis.model[coords] = x
        q = -analysis.Q()
        ret = [q.x, np.array(list(map(q.d, xs)))]
        return ret
        # logger.debug("\n" + np.array_str(model.y.astype(float), precision=2, max_line_width=100))
        # logger.debug("\n" + np.array_str(model.stepwise_values().astype(float), precision=2, max_line_width=100))
        # logger.debug((float(q), float(reg), float(q + reg)))
        # logger.debug("dq:\n" + np.array_str(np.array(list(map(q.d, xs))), max_line_width=100, precision=2))
        # logger.debug("dreg:\n" + np.array_str(np.array(list(map(reg.d, xs))), max_line_width=100, precision=2))

    def run(self, niter):
        self.update_observers('begin')
        model = self._analysis.model
        self._analysis.E_step()
        for i in range(niter):
            # Perform model optimization
            self.update_observers('pre-M step', i=i, niter=niter)
            coord_list = self._coordinates(i)
            for coords in coord_list:
                bounds = self._bounds(coords)
                x0 = model[coords]
                res = scipy.optimize.minimize(self._f, x0, 
                        args=(self._analysis, coords,), 
                        bounds=bounds, 
                        method="L-BFGS-B")
            self.update_observers('post-M step', results=res, i=i)
            # Perform E-step
            self.update_observers('pre-E step', i=i)
            self._analysis.E_step()
            self.update_observers('post-E step', i=i)
        # Conclude the optimization and perform any necessary callbacks.
        self.update_observers('optimization finished')

    def update_observers(self, *args, **kwargs):
        kwargs.update({'analysis': self._analysis})
        Observable.update_observers(self, *args, **kwargs)

## LISTENER CLASSES
class HiddenStateOccupancyPrinter(Observer):

    def update(self, message, *args, **kwargs):
        if message == "post-E step":
            hso = np.sum(kwargs['analysis']._im.xisums, axis=(0, 1))
            hso /= hso.sum()
            logger.debug("hidden state occupancy: %s", hso)

class ProgressPrinter(Observer):

    def update(self, message, *args, **kwargs):
        if message == "pre-M step":
            logger.info("Optimization iteration %d of %d...", kwargs['i'] + 1, kwargs['niter'])
        elif message == "pre-E step":
            logger.info("Running E-step...")
        elif message == "post-E step":
            logger.info("E-step completed.")

class LoglikelihoodPrinter(Observer):

    def update(self, message, *args, **kwargs):
        if message == "post-E step":
            ll = kwargs['analysis'].loglik()
            logger.info("Loglik: %f", ll)

class AnalysisSaver(Observer):

    def __init__(self, outdir):
        self._outdir = outdir

    def update(self, message, *args, **kwargs):
        dump = kwargs['analysis'].dump
        if message == "post-E step":
            dump(os.path.join(self._outdir, ".model.iter%d" % i))
        elif message == "optimization finished":
            dump(os.path.join(self._outdir, "model.iter%d" % i))

class ParameterOptimizer(Observer):

    def __init__(self, param, bounds):
        self._param = param
        self._bounds = bounds

    def update(self, message, *args, **kwargs):
        if message != "pre-M step":
            return
        param = self._param
        logger.debug("Updating %s", param)
        analysis = kwargs['analysis']
        if param not in ("theta", "rho"):
            raise RuntimeError("unrecognized param")
        x0 = getattr(analysis, param)
        logger.debug("Old %s: Q(%g)=%g", param, x0,
                     self._f(x0, analysis, param))
        res = scipy.optimize.minimize_scalar(self._f,
                                             args=(analysis, param),
                                             method='bounded',
                                             bounds=self._bounds)
        logger.debug("New %s: Q(%g)=%g", param, res.x, res.fun)
        setattr(analysis, param, res.x)

    def _f(self, x, analysis, param):
        setattr(analysis, param, x)
        # derivatives curretly not supported for 1D optimization. not
        # clear if they really help.
        return -float(analysis.Q()) 

class SMCPPOptimizer(AbstractOptimizer):
    'Model fitting for one population.'
    def __init__(self, analysis):
        AbstractOptimizer.__init__(self, analysis)
        self.register(LoglikelihoodPrinter())
        self.register(HiddenStateOccupancyPrinter())
        self.register(ProgressPrinter())

    def _coordinates(self):
        model = self._analysis.model
        return [list(range(b, min(model.K, b + self.block_size))) 
                for b in range(0, model.K - self.block_size + 1, self.block_size - 2)]

class TwoPopulationOptimizer(SMCPPOptimizer):

    def __init__(self, analysis, outdir, fix_rho, split_bounds=None):
        AnalysisOptimizer.__init__(analysis)
        if split_bounds is not None:
            self.register(ParameterOptimizer("split", split_bounds))

    def _coordinates(self):
        model = self._analysis.model
        c1 = [list(range(b, min(model.K, b + self.block_size))) 
                for b in range(0, model.K - self.block_size + 1, self.block_size - 2)]
        c2 = [tuple(a for a in cc if model.split_ind <= a) for cc in c1]
        return [(i, cc) for i, c in enumerate([c1, c2]) for cc in c]
