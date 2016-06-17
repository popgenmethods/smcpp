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
    def __init__(self, iserv, outdir):
        self._iserv = iserv
        self._bounds = iserv.bounds
        self._outdir = outdir
        logger.debug("Initializing model(s)")
        logger.debug("Performing initial E step")
        iserv.E_step()

    def run(self, niter, blocks, fix_rho):
        iserv = self._iserv
        models = iserv.model
        ll = sum([_ for _ in iserv.loglik()])
        reg = sum([m.regularizer() for m in models])
        llold = ll - reg
        logger.info("ll:%f reg:%f" % (ll, reg))
        logger.info("Starting loglik: %f" % llold)
        for i in range(niter):
            logger.info("EM iteration %d/%d" % (i + 1, niter))
            logger.info("\tM-step...")
            if not fix_rho:
                self._optimize_param("rho")
            logger.debug("starting model:\n%s\n%s" % (
                np.array_str(models[0].y.astype('float'), precision=3),
                np.array_str(models[0].stepwise_values().astype(
                    'float'), precision=3)))
            B = len(models[0].y) // blocks
            # for b in range(-1, blocks):
            for b in range(0, models[0].K, blocks - 2):
                self._coords = [(0, c) for c in range(b, min(models[0].K, b + blocks))]
                logger.info("optimizing coords:\n%s" % str(self._coords))
                self._optimize(models)
            # for v in [0, 1]:
            #     self._coords = [(mi, cc) for mi, m in enumerate(self._iserv.model) for cc in m.coords[v::2]]
            #     logger.info("optimizing coords:\n%s" % str(self._coords))
            #     self._optimize(models)
            logger.info("Current model(s):")
            for j, m in enumerate(models, 1):
                logger.info("Pop %d:\n%s" % (j, np.array_str(np.array(m.stepwise_values()).astype(float), precision=2)))
            iserv.model = models
            logger.info("\tE-step...")
            iserv.E_step()
            ll = np.sum(iserv.loglik())
            reg = sum([m.regularizer() for m in models])
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
        q = -np.mean(self._iserv.Q())
        logger.debug("f_%s: q(%f)=%f dq=%f" % (param, x0, q.x, q.d(x)))
        return (q.x, np.array([q.d(x)]))

    def _f(self, xs, models):
        xs = ad.adnumber(xs)
        yy = models[0].y.astype(object)
        for (xi, (_, cc)) in zip(xs, self._coords):
            yy[cc] = xi
        models[0].y = yy
        self._pre_Q(models)
        self._iserv.model = models
        q = -np.mean(self._iserv.Q())
        reg = np.mean([m.regularizer() for m in models])
        logger.debug("\n" + np.array_str(models[0].y.astype(float), precision=2, max_line_width=100))
        logger.debug("\n" + np.array_str(models[0].stepwise_values().astype(float), precision=2, max_line_width=100))
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
        # self._iserv.erivatives = [[d]] * self._npop
        x0 = getattr(self._iserv, param)
        logger.info("old %s: f(%g)=%g" % (param, x0, self._f_param([x0], param)[0]))
        bounds = [(1e-6, 1e-2)]
        ret = scipy.optimize.fmin_l_bfgs_b(self._f_param, x0, None, args=(param,), bounds=bounds, disp=False)
        x = ret[0].item()
        logger.info("new %s: f(%g)=%g" % (param, x, self._f_param([x], param)[0]))
        setattr(self._iserv, param, x)

    def _optimize(self, models):
        logger.debug("Performing a round of optimization")
        x0 = np.array([float(models[i].y[cc]) for i, cc in self._coords])
        # self._iserv.derivatives = [[(0, x) for x in range(len(models[0].s))]]
        if os.environ.get("SMCPP_GRADIENT_CHECK", False):
            logger.info("gradient check")
            f0, fp = self._f(x0, models)
            for i in range(len(x0)):
                x0c = x0.copy()
                x0c[i] += 1e-8
                f1, _ = self._f(x0c, models)
                logger.info((i, cc, f1, f0, (f1 - f0) / 1e-8, fp[i]))
        bounds = np.array([tuple(self._bounds[(0, cc)]) for i, cc in self._coords])
        # res = spg.SPG(
        #         lambda x: self._f(x, models), 
        #         lambda x: spg.projectBound(x, bounds[:, 0], bounds[:, 1]),
        #         x0)
        bounds = [(max(bd[0], 0.5 * xx0), min(bd[1], 2. * xx0)) for bd, xx0 in zip(bounds, x0)]
        # res = scipy.optimize.fmin_tnc(self._f, x0, None, args=[models], bounds=bounds, disp=5, xtol=1e-4)
        # eps = np.finfo(float).eps
        # bds = [[max(bb[0], xx - 5.), min(bb[1], xx + 5.)] for bb, xx in zip(bounds, x0)]
        res = scipy.optimize.fmin_l_bfgs_b(self._f, x0, None, args=[models], bounds=bounds)
        if res[2]['warnflag'] != 0:
            logger.warn(res[2])
        print(res)
        yy = models[0].y.copy()
        for (x, (_, i)) in zip(res[0], self._coords):
            yy[i] = x
        models[0].y = yy
        logger.info("new model: f(m)=%g" % res[1])
        self._post_optimize(models)
        self._iserv.model = models
        return res[1]

    def _pre_Q(self, models):
        pass

    def _post_optimize(self, models):
        pass
