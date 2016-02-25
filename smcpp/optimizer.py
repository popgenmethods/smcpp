import numpy as np
import scipy.optimize
import logging
import os.path
import ad

from . import estimation_tools

logger = logging.getLogger(__name__)

class PopulationOptimizer(object):
    def __init__(self, iserv, bounds, cmdargs):
        self._iserv = iserv
        self._bounds = bounds
        self._cmdargs = cmdargs
        self._precond = iserv.precond
        self._K = self._iserv.model[0].K
        self._coords = [(i, cc) for i, m in enumerate(self._iserv.model) for cc in m.coords]
        for m in self._iserv.model:
            if m.K != self._K:
                raise RuntimeError("models need to have same time periods and change points")

    def run(self, niter):
        iserv = self._iserv
        logger.debug("Initializing model(s)")
        models = iserv.model
        iserv.set_params(models, False)
        logger.debug("Performing initial E step")
        iserv.E_step()
        llold = np.mean([x for loglik in iserv.loglik() for x in loglik])
        logger.debug("Starting loglik: %g" % llold)
        for i in range(niter):
            logger.info("EM iteration %d/%d" % (i + 1, niter))
            logger.info("\tM-step...")
            self._optimize(models)
            logger.info("Current model(s):")
            for j, m in enumerate(models, 1):
                logger.info("Pop %d:\n%s" % (j, str(m.x[:2])))
            iserv.set_params(models, False)
            logger.info("\tE-step...")
            iserv.E_step()
            ll = np.mean([x for loglik in iserv.loglik() for x in loglik])
            logger.info("\tNew/old loglik: %f/%f" % (ll, llold))
            if ll < llold:
                logger.warn("Log-likelihood decreased")
            llold = ll
            iserv.dump([[os.path.join(self._cmdargs.outdir, ".pop%d.iter%d" % (j, i))] for j in range(len(models))])
        ## Optimization concluded
        iserv.dump([[os.path.join(self._cmdargs.outdir, ".pop%d.final" % j)] for j in range(len(models))])
        return llold

    def _f(self, xs, models):
        xs = ad.adnumber(xs)
        for i, (a, cc) in enumerate(self._coords):
            models[a][cc] = xs[i] * models[a].precond[cc]
        self._pre_Q(models)
        self._iserv.set_params(models, True)
        q = self._iserv.Q()
        ll = -np.mean(q)
        for m in models:
            ll += estimation_tools.regularizer(m, self._cmdargs.regularization_penalty)
        return [ll.x, np.array(list(map(ll.d, xs)))]

    def _optimize(self, models):
        logger.debug("Performing a round of optimization")
        # logger.info("gradient check")
        # model = models[0]
        # xx0 = np.array([model[cc] / model.precond[cc] for cc in model.coords])
        # f0, fp = self._f(xx0, models)
        # for i, cc in enumerate(model.coords):
        #     x0c = xx0.copy()
        #     x0c[i] += 1e-8
        #     f1, _ = self._f(x0c, models)
        #     logger.info((i, cc, f1, f0, (f1 - f0) / 1e-8, fp[i]))
        res = scipy.optimize.fmin_l_bfgs_b(self._f, 
                [models[i][cc] / models[i].precond[cc] for i, cc in self._coords],
                None,
                args=[models], bounds=[tuple(self._bounds[cc] / models[i].precond[cc]) for i, cc in self._coords],
                factr=1e9)
        for xx, (i, cc) in zip(res[0], self._coords):
            models[i][cc] = xx * models[i].precond[cc]
        self._post_optimize(models)

    def _pre_Q(self, models):
        pass

    def _post_optimize(self, models):
        pass

class TwoPopulationOptimizer(PopulationOptimizer):
    def _join_before_split(self, models):
        for a, cc in self._coords:
            if a == 0 and cc[1] >= self._split:
                models[1][cc] = models[0][cc]

    # Alias these methods to fix stuff before and after split
    _pre_Q = _post_optimize = _join_before_split

    def run(self, niter):
        upper = 2 * self._K
        lower = 0
        self._split = self._K
        self._old_aic = np.inf
        i = 1
        models = self._iserv.model
        while True:
            logger.info("Outer iteration %d / split point %d" % (i, self._split))
            self._coords = [(0, cc) for cc in models[0].coords]
            self._coords += [(1, cc) for cc in models[1].coords if cc[1] < self._split]
            ll = PopulationOptimizer.run(self, niter)
            aic = 2 * (len(self._coords) - ll)
            logger.info((len(self._coords), ll, aic))
            logger.info("AIC old/new: %g/%g" % (self._old_aic, aic))
            if aic < self._old_aic:
                upper = self._split
            else:
                lower = self._split
            self._old_aic = aic
            new_split = int(0.5 * (upper - lower))
            if new_split == self._split:
                break
            self._split = new_split
            i += 1
        logger.info("split chosen to be: " % self._split)
    
