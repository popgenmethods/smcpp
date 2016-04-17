from __future__ import absolute_import, division, print_function
import numpy as np
import scipy.optimize
import logging
import os.path
import ad

from . import estimation_tools

logger = logging.getLogger(__name__)

class PopulationOptimizer(object):
    _npop = 1
    def __init__(self, iserv, outdir):
        self._iserv = iserv
        self._bounds = iserv.bounds
        self._outdir = outdir
        self._precond = iserv.precond
        self._K = self._iserv.model[0].K
        self._coords = [(i, cc) for i, m in enumerate(self._iserv.model) for cc in m.coords]
        for m in self._iserv.model:
            if m.K != self._K:
                raise RuntimeError("models need to have same time periods and change points")
        logger.debug("Initializing model(s)")
        logger.debug("Performing initial E step")
        iserv.E_step()

    def run(self, niter, fix_rho):
        iserv = self._iserv
        models = iserv.model
        ll = np.sum([x for loglik in iserv.loglik() for x in loglik])
        reg = np.sum(self._iserv.penalize(models))
        llold = ll - reg
        logger.info("ll:%f reg:%f" % (ll, reg))
        logger.info("Starting loglik: %f" % llold)
        for i in range(niter):
            logger.info("EM iteration %d/%d" % (i + 1, niter))
            logger.info("\tM-step...")
            if not fix_rho:
                self._optimize_param("rho")
            logger.debug("starting model:\n%s" % np.array_str(models[0].x.astype(float), precision=2))
            self._optimize(models)
            logger.info("Current model(s):")
            for j, m in enumerate(models, 1):
                logger.info("Pop %d:\n%s" % (j, np.array_str(np.array(m.x[:2]).astype(float), precision=2)))
            iserv.model = models
            logger.info("\tE-step...")
            iserv.E_step()
            ll = np.sum([x for loglik in iserv.loglik() for x in loglik])
            reg = np.sum(iserv.penalize(models))
            logger.info("ll:%f reg:%f" % (ll, reg))
            ll -= reg
            logger.info("\tNew/old loglik: %f/%f" % (ll, llold))
            if ll < llold:
                logger.warn("Log-likelihood decreased")
            llold = ll
            iserv.dump([[os.path.join(self._outdir, ".pop%d.iter%d" % (j, i))] for j in range(len(models))])
        ## Optimization concluded
        iserv.dump([[os.path.join(self._outdir, "pop%d.final" % j)] for j in range(len(models))])
        return llold

    def _f_param(self, x, param):
        x0 = x[0]
        x = ad.adnumber(x0)
        setattr(self._iserv, param, x)
        q = -sum([u for l in self._iserv.Q() for u in l])
        logger.debug("f_%s: q(%f)=%f dq=%f" % (param, x0, q.x, q.d(x)))
        return (q.x, np.array([q.d(x)]))

    def _f(self, xs, models):
        xs = ad.adnumber(xs)
        for i, xx in enumerate(xs):
            xx.tag = i
        for i, (a, cc) in enumerate(self._coords):
            models[a][cc] = xs[i] * models[a].precond[cc]
        self._pre_Q(models)
        for m in models:
            logger.debug(m.x[:2].astype('float'))
        # for m in models:
        #     logger.debug("\n%s" % np.array_str(m.x[:2].astype(float), precision=4))
        self._iserv.model = models
        q = self._iserv.Q()
        reg = np.mean(self._iserv.penalize(models))
        ll = -np.mean([sum(qq) for qq in q])
        ll += reg
        ret = [ll.x, np.array(list(map(ll.d, xs)))]
        # logger.debug(ret[0])
        return ret

    def _optimize_param(self, param):
        logger.debug("Updating %s" % param)
        if param == "theta":
            d = (3, -1)
        elif param == "rho":
            d = (4, -1)
        else:
            raise RuntimeError("unrecognized param")
        self._iserv.derivatives = [[d]] * self._npop
        x0 = getattr(self._iserv, param)
        logger.info("old %s: f(%g)=%g" % (param, x0, self._f_param([x0], param)[0]))
        bounds = [(1e-6, 1e-2)]
        x = scipy.optimize.fmin_l_bfgs_b(self._f_param, x0, None, args=(param,), bounds=bounds, disp=False)[0].item()
        logger.info("new %s: f(%g)=%g" % (param, x, self._f_param([x], param)[0]))
        setattr(self._iserv, param, x)

    def _optimize(self, models):
        logger.debug("Performing a round of optimization")
        x0 = np.array([float(models[i][cc] / models[i].precond[cc]) for i, cc in self._coords])
        self._iserv.derivatives = [m.coords for m in models]
        # logger.info("gradient check")
        # f0, fp = self._f(x0, models)
        # for i in range(len(x0)):
        #     x0c = x0.copy()
        #     x0c[i] += 1e-8
        #     f1, _ = self._f(x0c, models)
        #     logger.info((i, cc, f1, f0, (f1 - f0) / 1e-8, fp[i]))
        # logger.info(scipy.optimize.check_grad(lambda x: self._f(x, models)[0], lambda x: self._f(x, models)[1], x0))
        bounds = [tuple(self._bounds[cc] / models[i].precond[cc]) for i, cc in self._coords]
        res = scipy.optimize.fmin_l_bfgs_b(self._f, x0, None, args=[models], bounds=bounds, factr=1e10)
        if res[2]['warnflag'] != 0:
            logger.warn(res[2])
        for xx, (i, cc) in zip(res[0], self._coords):
            models[i][cc] = xx * models[i].precond[cc]
        logger.info("new model: f(m)=%g" % res[1])
        self._post_optimize(models)
        self._iserv.model = models
        return res[1]

    def _pre_Q(self, models):
        pass

    def _post_optimize(self, models):
        pass

class TwoPopulationOptimizer(PopulationOptimizer):
    _npop = 2
    def _join_before_split(self, models):
        for a, cc in self._coords:
            if a == 0 and cc[1] >= self._split:
                models[1][cc] = models[0][cc]

    # Alias these methods to fix stuff before and after split
    _pre_Q = _post_optimize = _join_before_split

    def run(self, niter, fix_rho):
        upper = 2 * self._K
        lower = 0
        self._split = self._K
        self._old_aic = np.inf
        i = 1
        models = self._iserv.model
        cs = np.concatenate([np.cumsum(models[0][2]), [np.inf]])
        ll = None
        while True:
            logger.info("Outer iteration %d" % i)
            logger.info("split point %d:%g" % (self._split, cs[self._split]))
            self._iserv.reset()
            # Optimize the blocks of coords separately to speed things up
            for j in range(niter):
                logger.info("Pseudo-iteration %d/%d" % (j + 1, niter))
                for coords in [
                        [(0, cc) for cc in models[0].coords if cc[0] < self._split],
                        [(1, cc) for cc in models[1].coords if cc[1] < self._split] ]:
                    # reset models
                    self._coords = coords
                    ll = PopulationOptimizer.run(self, 1, fix_rho)
                    self._iserv.E_step()
            nc = len(models[0].coords)
            nc += sum([cc[1] < self._split for cc in models[1].coords])
            aic = 2 * (nc - ll)
            logger.info("AIC old/new: %g/%g" % (self._old_aic, aic))
            if aic < self._old_aic:
                upper = self._split
            else:
                lower = self._split
            self._old_aic = aic
            new_split = int(0.5 * (upper + lower))
            if abs(new_split - self._split) == 1:
                break
            self._split = new_split
            i += 1
        logger.info("split chosen to be: [%g, %g)" % (cs[self._split], cs[self._split + 1]))
