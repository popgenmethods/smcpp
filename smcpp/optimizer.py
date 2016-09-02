from __future__ import absolute_import, division, print_function
import numpy as np
import scipy.optimize
import os.path
import os
import ad
import ad.admath
import tempfile
import wrapt
import subprocess
import random
from six.moves import zip_longest
from abc import abstractmethod
import pprint

# some useful functions were added to shutil in Python 3.3
try:
    from shutil import which, get_terminal_size
except ImportError:
    from backports.shutil_which import which
    from backports.shutil_get_terminal_size import get_terminal_size

from . import logging, util, _smcpp
from .observe import targets, Observable, Observer

logger = logging.getLogger(__name__)


class EMTerminationException(Exception):
    "Thrown when EM algorithm reaches stopping criterion."
    pass


class AbstractOptimizer(Observable):
    '''
    Abstract representation of the execution flow of the optimizer.
    '''
    def __init__(self, analysis, algorithm, tolerance, solver_args={}):
        Observable.__init__(self)
        self._analysis = analysis
        self._algorithm = algorithm
        self._tolerance = tolerance
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

    def _f(self, x, analysis, coords, k=None):
        logger.debug(x.astype('float'))
        x = self._prepare_x(x)
        analysis.model[coords] = x
        q = -analysis.Q(k)
        ret = [q.x, np.array(list(map(q.d, x)))]
        return ret

    def _minimize(self, x0, coords, bounds):
        if os.environ.get("SMCPP_GRADIENT_CHECK", False):
            print("\n\ngradient check")
            y, dy = self._f(x0, self._analysis, coords)
            for i in range(len(x0)):
                x0[i] += 1e-8
                y1, _ = self._f(x0, self._analysis, coords)
                print("***grad", i, y1, (y1 - y) * 1e8, dy[i])
                x0[i] -= 1e-8
        return scipy.optimize.minimize(self._f, x0,
                                       jac=True,
                                       args=(self._analysis, coords),
                                       bounds=bounds,
                                       options=self._solver_args,
                                       method=self._algorithm)

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
                    bounds = self._bounds(coords)
                    x0 = self._analysis.model[coords]
                    res = self._minimize(x0, coords, bounds)
                    self._analysis.model[coords] = res.x
                    self.update_observers('post mini M-step',
                                          coords=coords,
                                          res=res, **kwargs)
                self.update_observers('post M-step', **kwargs)
        except EMTerminationException:
            pass
        # Conclude the optimization and perform any necessary callbacks.
        self.update_observers('optimization finished')

    def update_observers(self, *args, **kwargs):
        kwargs.update({
            'optimizer': self,
            'analysis': self._analysis,
            'model': self._analysis.model})
        Observable.update_observers(self, *args, **kwargs)

# LISTENER CLASSES

class HiddenStateOccupancyPrinter(Observer):

    @targets("post E-step")
    def update(self, message, *args, **kwargs):
        hso = np.sum(
                [np.sum(im.getXisums(), axis=(0, 1))
                    for im in kwargs['analysis']._ims.values()], axis=0)
        hso /= hso.sum()
        logger.debug("hidden state occupancy:\n%s",
                     np.array_str(hso, precision=2))


class ProgressPrinter(Observer):

    def update(self, message, *args, **kwargs):
        if message == "begin":
            logger.info("Starting EM algorithm...")
        if message == "pre E-step":
            logger.info("EM iteration %d of %d...",
                        kwargs['i'] + 1, kwargs['niter'])
        if message == "M step":
            logger.debug("Optimizing coordinates %s", kwargs['coords'])
        if message == "post mini M-step":
            logger.debug("Results of optimizer:\n%s", kwargs['res'])
        if message == "post M-step":
            logger.info("Current model:\n%s", kwargs['model'].to_s())

class LoglikelihoodMonitor(Observer):

    def __init__(self):
        self._old_loglik = None

    @targets("post E-step")
    def update(self, message, *args, **kwargs):
        ll = kwargs['analysis'].loglik()
        if self._old_loglik is None:
            logger.info("Loglik: %f", ll)
        else:
            improvement = (self._old_loglik - ll) / self._old_loglik
            logger.info("New loglik: %f\t(old: %f [%f%%])",
                    ll, self._old_loglik, 100. * improvement)
            tol = kwargs['optimizer']._tolerance
            if improvement < 0:
                logger.warn("Loglik decreased")
            elif improvement < tol:
                logger.info("Log-likelihood improvement < tol=%g; terminating", tol)
                raise EMTerminationException()
        self._old_loglik = ll


class ModelPrinter(Observer):

    @targets("post M-step")
    def update(self, message, *args, **kwargs):
        logger.info("Model: %s", kwargs['model'].to_s())


class AnalysisSaver(Observer):

    def __init__(self, outdir):
        self._outdir = outdir

    def update(self, message, *args, **kwargs):
        dump = kwargs['analysis'].dump
        if message == "post E-step":
            i = kwargs['i']
            dump(os.path.join(self._outdir, ".model.iter%d" % i))
        elif message == "optimization finished":
            dump(os.path.join(self._outdir, "model.final"))


class KnotOptimizer(Observer):

    @targets("pre M-step")
    def update(self, message, *args, **kwargs):
        # pick a random knot and optimize
        model = kwargs['model']
        knots = model._knots
        inds = np.arange(1, len(knots) - 1)  # hold first and last knots fixed
        for i in np.random.choice(inds, size=int(len(inds) * .2), replace=False):
            if i == 0:
                bounds = (1e-6, 0.9 * knots[1])
            elif i == len(knots) - 1:
                bounds = (1.1 * knots[-2], 2 * knots[-1])
            else:
                bounds = (knots[i - 1] * 1.1, knots[i + 1] * 0.9)
            analysis = kwargs['analysis']
            opt = kwargs['optimizer']
            bounds = (bounds, opt._bounds([i])[0])
            x0 = (knots[i], model[i])
            logger.info("Old knot %d=(%f,%f) Q=%f", i, x0[0], x0[1], self._f(x0, analysis, i))
            logger.debug("Bounds: %s", bounds)
            res = scipy.optimize.minimize(self._f,
                                          x0=x0,
                                          args=(analysis, i),
                                          bounds=bounds)
            logger.info("New knot %d=(%f,%f) Q=%f", i, res.x[0], res.x[1], res.fun)
            knots[i] = res.x[0]
            model[i] = res.x[1]
            model.refit()

    def _f(self, x, analysis, i):
        analysis.model._knots[i] = x[0]
        analysis.model[i] = x[1]
        # derivatives curretly not supported for 1D optimization. not
        # clear if they really help.
        ret = -float(analysis.Q())
        logger.debug("knot %d Q(%s)=%f", i, x, ret)
        return ret


class ParameterOptimizer(Observer):

    def __init__(self, param, bounds, target="analysis"):
        self._param = param
        self._bounds = bounds
        self._target = target

    @targets("pre M-step")
    def update(self, message, *args, **kwargs):
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


class SplineDumper(Observer):

    def __init__(self, outdir):
        self._outdir = outdir

    @targets("post M-step")
    def update(self, message, *args, **kwargs):
        with open(os.path.join(self._outdir, ".spline.txt"), "wt") as f:
            kwargs['model']._spline.dump(f)


class AsciiPlotter(Observer):

    def __init__(self, gnuplot_path):
        self._gnuplot_path = gnuplot_path

    @targets(["post M-step", "post mini M-step"])
    def update(self, message, *args, **kwargs):
        model = kwargs['model']
        two_pop = hasattr(model, 'split')
        if two_pop:
            # plot split models
            x = np.cumsum(model.model1.s)
            y = model.model1.stepwise_values()
            z = model.model2.stepwise_values()
            data = "\n".join([",".join(map(str, row)) for row in zip(x, y)])
            data += "\n" * 3
            data += "\n".join([",".join(map(str, row)) for row in zip(x, z) if row[0] <= model.split])
        else:
            x = np.cumsum(model.s)
            y = model.stepwise_values()
            u = model._knots
            v = np.exp(model[:].astype('float'))
            data = "\n".join([",".join(map(str, row)) for row in zip(x, y)])
            data += "\n" * 3
            data += "\n".join([",".join(map(str, row)) for row in zip(u, v)])
        # Fire up the plot process and let'ter rip.
        gnuplot = subprocess.Popen([self._gnuplot_path],
                                   stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE)
        def write(x):
            x += "\n"
            gnuplot.stdin.write(x.encode())
        columns, lines = np.maximum(get_terminal_size(), [80, 25])
        width = columns * 3 // 5
        height = 25
        write("set term dumb {} {}".format(width, height))
        write("set datafile separator \",\"")
        write("set xlabel \"Time\"")
        write("set ylabel \"N0\"")
        write("set logscale xy")
        with tempfile.NamedTemporaryFile("wt") as f:
            plot_cmd = "plot '%s' i 0 with lines title 'Pop. 1'" % f.name
            if two_pop:
                plot_cmd += ", '' i 1 with lines title 'Pop. 2';"
            else:
                plot_cmd += ", '' i 1 with points notitle;"
            write(plot_cmd)
            open(f.name, "wt").write(data)
            write("unset key")
            write("exit")
            (stdout, stderr) = gnuplot.communicate()
            graph = stdout.decode()
        logfun = logger.debug if message == "post mini M-step" else logger.info
        logfun("Plot of current model:\n%s", graph)


# class TransitionDebug(Observer):
# 
#     def __init__(self, path):
#         self._path = path
#         try:
#             os.makedirs(path)
#         except OSError:
#             pass
# 
#     @targets("post mini M-step")
#     def update(self, message, *args, **kwargs):
#         im = kwargs['analysis']._im
#         T = im.transition
#         xis = np.sum(im.xisums, axis=0)
#         np.savetxt(os.path.join(self._path, "xis.txt"),
#                    xis.astype("float"), fmt="%g")
#         log_T = np.array(ad.admath.log(T))
#         np.savetxt(os.path.join(self._path, "log_T.txt"),
#                    log_T.astype("float"), fmt="%g")
#         q3 = log_T * xis
#         np.savetxt(os.path.join(self._path, "q3.txt"),
#                    q3.astype("float"), fmt="%g")
#         for i, d in enumerate(im.model.dlist):
#             f = np.vectorize(lambda x, d=d: x.d(d))
#             np.savetxt(os.path.join(self._path, "q3.%d.txt" % i),
#                        f(q3).astype("float"), fmt="%g")


class SMCPPOptimizer(AbstractOptimizer):
    'Model fitting for one population.'

    def __init__(self, analysis, algorithm, tolerance, solver_args):
        AbstractOptimizer.__init__(self, analysis, algorithm, tolerance, solver_args)
        observers = [
            HiddenStateOccupancyPrinter(),
            ProgressPrinter(),
            ModelPrinter(),
            LoglikelihoodMonitor(),
            # TransitionDebug("/export/home/terhorst/Dropbox.new/Dropbox/tdtmp"),
            # SplineDumper("/export/home/terhorst/Dropbox.new/Dropbox/tdtmp")
        ]
        gnuplot = which("gnuplot")
        if gnuplot:
            observers.append(AsciiPlotter(gnuplot))
        for obs in observers:
            self.register(obs)

    def _coordinates(self):
        model = self._analysis.model
        ret = []
        for b in range(0, model.K - self.block_size + 1, max(1, self.block_size - 2)):
            ret.append(list(range(b, min(model.K, b + self.block_size))))
        # After all coordinate-wise updates, optimize over whole function
        ret = ret[::-1]
        ret.append(list(range(model.K)))
        return ret

    def _bounds(self, coords):
        ret = np.log([self._analysis._bounds] * len(coords))
        return ret


class TwoPopulationOptimizer(SMCPPOptimizer):
    'Model fitting for two populations.'

    def _coordinates(self):
        K = len(self._analysis.model.model1[:])
        c1, c2 = [[list(range(b, min(K, b + self.block_size)))
            for b in range(0, ub + 1, self.block_size - 2)][::-1]
            for ub in [K - self.block_size, self._analysis.model.split_ind]]
        ret =  sorted([(i, cc) for i, c in enumerate([c1, c2]) for cc in c], key=lambda x: x[1])[::-1]
        return ret

    def _bounds(self, coords):
        return SMCPPOptimizer._bounds(self, coords[1])

class SplitOptimizer(TwoPopulationOptimizer):
    def _coordinates(self):
        K = self._analysis.model.distinguished_model(0).K
        return [(i, np.arange(K)) for i in (0, 1)]
