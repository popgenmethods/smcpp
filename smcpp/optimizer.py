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

    # In the one population case, this method adds derivative information to x
    def _prepare_x(self, x):
        return ad.adnumber(x)

    def _f(self, x, analysis, coords):
        x = self._prepare_x(x)
        analysis.model.reset_derivatives()
        analysis.model[coords] = x
        q = -analysis.Q()
        ret = [q.x, np.array(list(map(q.d, x)))]
        return ret
        # logger.debug("\n" + np.array_str(model.y.astype(float), precision=2, max_line_width=100))
        # logger.debug("\n" + np.array_str(model.stepwise_values().astype(float), precision=2, max_line_width=100))
        # logger.debug((float(q), float(reg), float(q + reg)))
        # logger.debug("dq:\n" + np.array_str(np.array(list(map(q.d, xs))), max_line_width=100, precision=2))
        # logger.debug("dreg:\n" + np.array_str(np.array(list(map(reg.d, xs))), max_line_width=100, precision=2))

    def _minimize(self, x0, coords, bounds):
        return scipy.optimize.minimize(self._f, x0,
                                       jac=True,
                                       args=(self._analysis, coords,),
                                       bounds=bounds,
                                       method="L-BFGS-B")

    def run(self, niter):
        self.update_observers('begin')
        model = self._analysis.model
        self._analysis.E_step()
        for i in range(niter):
            # Perform model optimization
            self.update_observers('pre-M step', i=i, niter=niter)
            coord_list = self._coordinates()
            for coords in coord_list:
                self.update_observers('M step', coords=coords)
                bounds = self._bounds(coords)
                x0 = model[coords]
                res = self._minimize(x0, coords, bounds)
                model[coords] = res.x
            self.update_observers('post-M step', results=res, i=i)
            # Perform E-step
            self.update_observers('pre-E step', i=i)
            self._analysis.E_step()
            self.update_observers('post-E step', i=i)
        # Conclude the optimization and perform any necessary callbacks.
        self.update_observers('optimization finished')

    def update_observers(self, *args, **kwargs):
        kwargs.update({'analysis': self._analysis, 'model': self._analysis.model})
        Observable.update_observers(self, *args, **kwargs)

## LISTENER CLASSES
class HiddenStateOccupancyPrinter(Observer):

    def update(self, message, *args, **kwargs):
        if message == "post-E step":
            hso = np.sum(kwargs['analysis']._im.xisums, axis=(0, 1))
            hso /= hso.sum()
            logger.debug("hidden state occupancy:\n%s", np.array_str(hso, precision=2))

class ProgressPrinter(Observer):

    def update(self, message, *args, **kwargs):
        if message == "begin":
            logger.info("Starting optimizer...")
        if message == "pre-M step":
            logger.info("Optimization iteration %d of %d...",
                        kwargs['i'] + 1, kwargs['niter'])
        if message == "M step":
            logger.debug("Optimizing coordinates %s", kwargs['coords'])

class LoglikelihoodPrinter(Observer):

    def update(self, message, *args, **kwargs):
        if message == "post-E step":
            ll = kwargs['analysis'].loglik()
            logger.info("Loglik: %f", ll)

class ModelPrinter(Observer):
    def update(self, message, *args, **kwargs):
        if message == "post-E step":
            logger.info("Model: %s", kwargs['model'].to_s())


class AnalysisSaver(Observer):

    def __init__(self, outdir):
        self._outdir = outdir

    def update(self, message, *args, **kwargs):
        dump = kwargs['analysis'].dump
        if message == "post-E step":
            i = kwargs['i']
            dump(os.path.join(self._outdir, ".model.iter%d" % i))
        elif message == "optimization finished":
            dump(os.path.join(self._outdir, "model.final"))

class ParameterOptimizer(Observer):

    def __init__(self, param, bounds, target="analysis"):
        self._param = param
        self._bounds = bounds
        self._target = target

    def update(self, message, *args, **kwargs):
        if message != "pre-M step":
            return
        param = self._param
        logger.debug("Updating %s", param)
        tgt = kwargs[self._target]
        analysis = kwargs['analysis']
        if param not in ("theta", "rho", "split"):
            raise RuntimeError("unrecognized param")
        x0 = getattr(tgt, param)
        logger.debug("Old %s: Q(%f)=%f", param, x0,
                     self._f(x0, analysis, tgt, param))
        res = scipy.optimize.minimize_scalar(self._f,
                                             args=(analysis, tgt, param),
                                             method='bounded',
                                             bounds=self._bounds)
        logger.debug("New %s: Q(%g)=%g", param, res.x, res.fun)
        setattr(tgt, param, res.x)

    def _f(self, x, analysis, tgt, param):
        setattr(tgt, param, x)
        # derivatives curretly not supported for 1D optimization. not
        # clear if they really help.
        ret = -float(analysis.Q()) 
        logger.debug("%s f(%f)=%f", param, x, ret)
        return ret

class SMCPPOptimizer(AbstractOptimizer):
    'Model fitting for one population.'

    def __init__(self, analysis):
        AbstractOptimizer.__init__(self, analysis)
        self.register(LoglikelihoodPrinter())
        self.register(HiddenStateOccupancyPrinter())
        self.register(ProgressPrinter())
        self.register(ModelPrinter())

    def _coordinates(self):
        model = self._analysis.model
        return [list(range(b, min(model.K, b + self.block_size))) 
                for b in range(0, model.K - self.block_size + 1, self.block_size - 2)]
    
    def _bounds(self, coords):
        return np.log([self._analysis._bounds] * len(coords))
    

class TwoPopulationOptimizer(SMCPPOptimizer):
    'Model fitting for two populations.'

    def __init__(self, analysis):
        SMCPPOptimizer.__init__(self, analysis)

    def _coordinates(self):
        K = self._analysis.model.distinguished_model.K
        c1, c2 = [[list(range(b, min(K, b + self.block_size)))
                   for b in range(0, ub + 1, self.block_size - 2)]
                  for ub in [K - self.block_size,
                             self._analysis.model.split_ind]]
        return [(i, cc) for i, c in enumerate([c1, c2]) for cc in c]

    def _minimize(self, x0, coords, bounds):
        return scipy.optimize.minimize(self._f, x0,
                                       jac=False,
                                       args=(self._analysis, coords,),
                                       bounds=bounds,
                                       method="L-BFGS-B")

    # Don't use derivatives in 2-pop case
    def _f(self, x, analysis, coords):
        return SMCPPOptimizer._f(self, x, analysis, coords)[0]

    def _bounds(self, coords):
        return SMCPPOptimizer._bounds(self, coords[1])

    # In the two population case, this method doesn't add derivatives since
    # we don't support them.
    def _prepare_x(self, x):
        return x
