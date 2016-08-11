from __future__ import absolute_import, division, print_function
import numpy as np
import scipy.optimize
import os.path
import os
import ad
import ad.admath
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

from . import logging, util
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
        return [ad.adnumber(xx, tag=i) for i, xx in enumerate(x)]

    def _f(self, x, analysis, coords, k=None):
        logger.debug(x.astype('float'))
        x = self._prepare_x(x)
        analysis.model.reset_derivatives()
        analysis.model[coords] = x
        q = -analysis.Q(k)
        ret = [q.x, np.array(list(map(q.d, x)))]
        return ret

    def _minimize(self, x0, coords, bounds):
        # print("\n\ngradient check")
        # for k in range(3):
        #     y, dy = self._f(x0, self._analysis, coords, k)
        #     for i in range(len(x0)):
        #         x0[i] += 1e-8
        #         y1, _ = self._f(x0, self._analysis, coords, k)
        #         print("***grad", k, i, y1, (y1 - y) * 1e8, dy[i])
        #         x0[i] -= 1e-8
        return scipy.optimize.minimize(self._f, x0,
                                       jac=True,
                                       args=(self._analysis, coords),
                                       bounds=bounds,
                                       # options={'xtol': .001},
                                       method="TNC")
        # return estimation_tools.adagrad(self._f, x0, bounds,
        #     stepsize=1.0,
        #     args=(self._analysis, coords))

    def run(self, niter):
        self.update_observers('begin')
        model = self._analysis.model
        for i in range(niter):
            # Perform E-step
            kwargs = {'i': i, 'niter': niter}
            self.update_observers('pre-E step', **kwargs)
            self._analysis.E_step()
            self.update_observers('post E-step', **kwargs)
            # Perform M-step
            self.update_observers('pre M-step', **kwargs)
            coord_list = self._coordinates()
            for coords in coord_list:
                self.update_observers('M step', coords=coords, **kwargs)
                bounds = self._bounds(coords)
                x0 = model[coords]
                res = self._minimize(x0, coords, bounds)

                # from collections import defaultdict
                # ep = self._analysis._im.emission_probs
                # gs = self._analysis._im.gamma_sums
                # M = len(next(iter(gs[0].values())))
                # ggs = defaultdict(lambda: np.zeros(M))
                # for g in gs:
                #     for k in g:
                #         ggs[k] += g[k]
                # pprint.pprint({(0, 0,0): ggs[(0, 0, 0)], (1,0,0): ggs[(1,0,0)]})
                # vv = {k: (ggs[k] * ad.admath.log(ep[k])) for k in ggs}
                # vi = sorted(vv.items(), key=lambda tup: abs(tup[1].sum()))
                # kept = vi[-5:]
                # pprint.pprint([(k, v.astype('float')) for k, v in kept])
                # try:
                #     pprint.pprint([(k, np.array([x.d(model.dlist[0])
                #                                  for x in v]))
                #                    for k, v in kept])
                # except:
                #     pass
                # vv = {k: (vv[k].x, [vv[k].d(l) for l in model.dlist]) for k in vv}
                # pprint.pprint(vv)

                self.update_observers('post mini M-step',
                                      coords=coords,
                                      res=res, **kwargs)
                model[coords] = res.x
            self.update_observers('post M-step', **kwargs)
        # Conclude the optimization and perform any necessary callbacks.
        self.update_observers('optimization finished')

    def update_observers(self, *args, **kwargs):
        kwargs.update({'analysis': self._analysis,
                       'model': self._analysis.model})
        Observable.update_observers(self, *args, **kwargs)

# LISTENER CLASSES

# Decorator to target specific messages.
def targets(target_message):
    @wrapt.decorator
    def wrapper(wrapped, instance, args, kwargs):
        message = args[0]
        if message == target_message:
            wrapped(instance, *args, **kwargs)
    return wrapper


class HiddenStateOccupancyPrinter(Observer):

    @targets("post E-step")
    def update(self, message, *args, **kwargs):
        hso = np.sum(kwargs['analysis']._im.xisums, axis=(0, 1))
        hso /= hso.sum()
        logger.debug("hidden state occupancy:\n%s",
                     np.array_str(hso, precision=2))


class ProgressPrinter(Observer):

    def update(self, message, *args, **kwargs):
        if message == "begin":
            logger.info("Starting optimizer...")
        if message == "pre E-step":
            logger.info("Optimization iteration %d of %d...",
                        kwargs['i'] + 1, kwargs['niter'])
        if message == "M step":
            logger.debug("Optimizing coordinates %s", kwargs['coords'])
        if message == "post mini M-step":
            logger.debug("Results of optimizer:\n%s", kwargs['res'])
        if message == "post M-step":
            logger.info("Current model:\n%s", kwargs['model'].to_s())


class LoglikelihoodPrinter(Observer):

    def __init__(self):
        self._old_loglik = None

    @targets("post E-step")
    def update(self, message, *args, **kwargs):
        ll = kwargs['analysis'].loglik()
        if self._old_loglik is None:
            logger.info("Loglik: %f", ll)
        else:
            logger.info("New loglik: %f\t(old loglik: %f)", ll, self._old_loglik)
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
        # Hold first and last knots fixed?
        i = np.random.choice(len(knots) - 1) + 1
        if i == 0:
            bounds = (0., 0.9 * knots[2])
        elif i == len(knots) - 1:
            bounds = (1.1 * knots[-2], 2 * knots[-1])
        else:
            bounds = (knots[i - 1] * 1.1, knots[i + 1] * 0.9)
        logger.debug("Updating knot %d=%s", i, knots[i])
        logger.debug("Bounds: (%f, %f)", *bounds)
        analysis = kwargs['analysis']
        logger.debug("Start: Q=%f", self._f(knots[i], analysis, i))
        res = scipy.optimize.minimize_scalar(self._f,
                                             args=(analysis, i),
                                             method='bounded',
                                             bounds=bounds)
        logger.debug("End: knot=%f Q=%f", res.x, res.fun)
        knots[i] = res.x
        model.refit()

    def _f(self, x, analysis, i):
        analysis.model._knots[i] = x
        analysis.model[:] = analysis.model[:].astype('float')
        analysis.model.refit()
        # derivatives curretly not supported for 1D optimization. not
        # clear if they really help.
        ret = -float(analysis.Q())
        logger.debug("knot %d Q(%f)=%f", i, x, ret)
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

    @targets("post M-step")
    def update(self, message, *args, **kwargs):
        model = kwargs['model']
        two_pop = hasattr(model, 'split')
        if two_pop:
            # plot split models
            x = np.cumsum(model.model1.s)
            y = model.model1.stepwise_values()
            z = model.model2.stepwise_values()
            ind = np.searchsorted(np.cumsum(x), model.split)
            z = z[:ind + 1]
            data = "\n".join([",".join(map(str, row)) for row in zip_longest(x, y, z, fillvalue=".")])
        else:
            x = np.cumsum(model.s)
            y = model.stepwise_values()
            data = "\n".join([",".join(map(str, row)) for row in zip(x, y)])
        # Fire up the plot process and let'ter rip.
        gnuplot = subprocess.Popen([self._gnuplot_path],
                                   stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE)
        def write(x):
            x += "\n"
            gnuplot.stdin.write(x.encode())
        columns, lines = get_terminal_size((80, 20))
        width = columns * 3 // 5
        height = 25
        write("set term dumb {} {}".format(width, height))
        write("set datafile separator \",\"")
        write("set xlabel \"Time\"")
        write("set ylabel \"N0\"")
        write("set logscale xy")
        plot_cmd = "plot '-' using 1:2 with lines title 'Pop. 1'"
        if two_pop:
            plot_cmd += ", '' using 1:3 with lines title 'Pop. 2';"
        write(plot_cmd)
        write(data)
        write("e")
        if two_pop:
            write(data)
            write("e")
        else:
            write("unset key")
        write("exit")
        (stdout, stderr) = gnuplot.communicate()
        graph = stdout.decode()
        logger.info("Current model:\n%s", graph)


class TransitionDebug(Observer):

    def __init__(self, path):
        self._path = path
        try:
            os.makedirs(path)
        except OSError:
            pass

    @targets("post mini M-step")
    def update(self, message, *args, **kwargs):
        im = kwargs['analysis']._im
        T = im.transition
        xis = np.sum(im.xisums, axis=0)
        np.savetxt(os.path.join(self._path, "xis.txt"),
                   xis.astype("float"), fmt="%g")
        log_T = np.array(ad.admath.log(T))
        np.savetxt(os.path.join(self._path, "log_T.txt"),
                   log_T.astype("float"), fmt="%g")
        q3 = log_T * xis
        np.savetxt(os.path.join(self._path, "q3.txt"),
                   q3.astype("float"), fmt="%g")
        for i, d in enumerate(im.model.dlist):
            f = np.vectorize(lambda x, d=d: x.d(d))
            np.savetxt(os.path.join(self._path, "q3.%d.txt" % i),
                       f(q3).astype("float"), fmt="%g")


class SMCPPOptimizer(AbstractOptimizer):
    'Model fitting for one population.'

    def __init__(self, analysis):
        AbstractOptimizer.__init__(self, analysis)
        observers = [
            LoglikelihoodPrinter(),
            HiddenStateOccupancyPrinter(),
            ProgressPrinter(),
            ModelPrinter(),
            KnotOptimizer(),
            TransitionDebug("/export/home/terhorst/Dropbox.new/Dropbox/tdtmp"),
            SplineDumper("/export/home/terhorst/Dropbox.new/Dropbox/tdtmp")
        ]
        gnuplot = which("gnuplot")
        if gnuplot:
            observers.append(AsciiPlotter(gnuplot))
        for obs in observers:
            self.register(obs)

    def _coordinates(self):
        model = self._analysis.model
        ret = []
        for b in range(model.K - self.block_size + 1):
            ret.append(list(range(b, min(model.K, b + self.block_size))))
        # After all coordinate-wise updates, optimize over whole function
        # ret = ret[::-1]
        ret.append(list(range(model.K)))
        return [random.choice(ret)]

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

    def _bounds(self, coords):
        return SMCPPOptimizer._bounds(self, coords[1])
