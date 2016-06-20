from __future__ import absolute_import, division, print_function
import numpy as np
import scipy.optimize
import logging
import os.path, os
import ad

from . import estimation_tools, spg

logger = logging.getLogger(__name__)

class PopulationOptimizer(object):
    _npop = 1
    def __init__(self, population, outdir, regularization_penalty):
        self._pop = population
        self._outdir = outdir
        self._penalty = regularization_penalty
        logger.debug("Initializing model")
        logger.debug("Performing initial E step")
        self._pop.E_step()

    def run(self, niter, blocks, fix_rho):
        pop = self._pop
        model = pop.model
        ll = pop.loglik()
        reg = model.regularizer()
        llold = ll - self._penalty * reg
        logger.info("ll:%f reg:%f" % (ll, reg))
        logger.info("Starting loglik: %f" % llold)
        for i in range(niter):
            logger.info("EM iteration %d/%d" % (i + 1, niter))
            logger.info("\tM-step...")
            if not fix_rho:
                self._optimize_param("rho")
            logger.debug("starting model:\n%s\n%s" % (
                np.array_str(model.y.astype('float'), precision=3),
                np.array_str(model.stepwise_values().astype(
                    'float'), precision=3)))
            B = len(model.y) // blocks
            # for b in range(-1, blocks):
            for b in range(0, model.K, blocks - 2):
                self._coords = list(range(b, min(model.K, b + blocks)))
                logger.info("optimizing coords:\n%s" % str(self._coords))
                self._optimize()
            logger.info("Current model:\n%s" % 
                    np.array_str(np.array(model.stepwise_values()).astype(float), precision=2))
            logger.info("\tE-step...")
            self._pop.E_step()
            ll = self._pop.loglik()
            reg = model.regularizer()
            logger.info("ll:%f reg:%f" % (ll, reg))
            ll -= self._penalty * reg
            logger.info("\tNew/old loglik: %f/%f" % (ll, llold))
            if ll < llold:
                logger.warn("Log-likelihood decreased")
            llold = ll
            self._pop.dump(os.path.join(self._outdir, ".pop0.iter%d" % i))
        ## Optimization concluded
        self._pop.dump(os.path.join(self._outdir, "pop0.final"))
        return llold

    def _f_param(self, x, param):
        setattr(self._pop, param, x)
        q = -self._pop.Q()
        logger.debug("f_%s: q(%f)=%f" % (param, x, q))
        return q

    def _f(self, xs):
        model = self._pop.model
        xs = ad.adnumber(xs)
        model[self._coords] = xs
        q = -self._pop.Q()
        reg = model.regularizer()
        logger.debug("\n" + np.array_str(model.y.astype(float), precision=2, max_line_width=100))
        logger.debug("\n" + np.array_str(model.stepwise_values().astype(float), precision=2, max_line_width=100))
        logger.debug((float(q), float(reg), float(q + reg)))
        logger.debug("dq:\n" + np.array_str(np.array(list(map(q.d, xs))), max_line_width=100, precision=2))
        logger.debug("dreg:\n" + np.array_str(np.array(list(map(reg.d, xs))), max_line_width=100, precision=2))
        q += reg
        ret = [q.x, np.array(list(map(q.d, xs)))]
        return ret

    def _optimize_param(self, param):
        logger.debug("Updating %s" % param)
        if param == "theta":
            d = (3, -1)
        elif param == "rho":
            d = (4, -1)
        else:
            raise RuntimeError("unrecognized param")
        x0 = getattr(self._pop, param)
        logger.info("old %s: f(%g)=%g" % (param, x0, self._f_param(x0, param)))
        bounds = (1e-6, 1e-2)
        res = scipy.optimize.minimize_scalar(self._f_param, args=(param,), method='bounded', bounds=bounds)
        logger.info("new %s: f(%g)=%g" % (param, res.x, res.fun))
        setattr(self._pop, param, res.x)

    def _optimize(self):
        model = self._pop.model
        logger.debug("Performing a round of optimization")
        x0 = model[self._coords].astype('float')
        if os.environ.get("SMCPP_GRADIENT_CHECK", False):
            logger.info("gradient check")
            f0, fp = self._f(x0)
            for i in range(len(x0)):
                x0c = x0.copy()
                x0c[i] += 1e-8
                f1, _ = self._f(x0c)
                logger.info((i, f1, f0, (f1 - f0) / 1e-8, fp[i]))
        # bounds = np.array([tuple(self._bounds[(0, cc)]) for i, cc in self._coords])
        # bounds = [(max(bd[0], 0.5 * xx0), min(bd[1], 2. * xx0)) for bd, xx0 in zip(bounds, x0)]
        bounds = np.log(self._pop.bounds[0, self._coords])
        res = scipy.optimize.minimize(self._f, x0, jac=True, bounds=bounds)
        if res.status != 0:
            logger.warn(res.message)
        # if res[2]['warnflag'] != 0:
            # logger.warn(res[2])
        model[self._coords] = res.x
        return res.fun
