import numpy as np
import scipy.optimize
import logging
import os.path

logger = logging.getLogger(__name__)

class _optimizer(object):
    def __init__(self, iserv, bounds, cmdargs):
        self._iserv = iserv
        self._bounds = bounds
        self._cmdargs = cmdargs
        self._coords = iserv.coords
        self._precond = iserv.precond
        self._K = self._iserv.model[0].K
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
        llold = sum(sum(ll) for ll in iserv.loglik())
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
            ll = sum(sum(ll) for ll in iserv.loglik())
            logger.info("\tNew/old loglik: %f/%f" % (ll, llold))
            if ll < llold:
                logger.warn("Log-likelihood decreased")
            llold = ll
            iserv.dump([os.path.join(self._cmdargs.outdir, ".pop%d.iter%d" % (j, i)) for j in range(len(models))])
        ## Optimization concluded
        iserv.dump([os.path.join(self._cmdargs.outdir, ".pop%d.final" % j) for j in range(len(models))])
        return llold

    def _f(self, xs):

        for xxs in xs:
            xs
        models = self._iserv.model
        regs = []
        for m, xx, coords, precond in zip(models, xs, self._coords, self._precond):
            for cc in coords:
                m.x[cc] = xx[cc] * precond[cc]
            m.flatten()
            regs.append(m.regularizer(self._cmdargs.lambda_penalty))
        self._iserv.set_params(models, self._coords)
        ret = []
        qq = self._iserv.Q()
        for q, reg, coords, precond in zip(qq, regs, self._coords, self._precond):
            lls, jacs = zip(*q)
            tmp = [-np.mean(lls), -np.mean(jacs, axis=0)]
            tmp[0] += reg[0]
            tmp[1] += reg[1]
            for i, cc in enumerate(coords):
                tmp[1][i] *= precond[cc]
            ret.append(tmp)
        return ret

class SinglePopulationOptimizer(_optimizer):
    def _f(self, xs):
        new_x = np.zeros([1, 2, self._K])
        coords = self._coords[0]
        new_x.__setitem__([0] + list(zip(*coords)), xs)
        return _optimizer._f(self, new_x)[0]

    def _optimize(self, models):
        logger.debug("Performing a round of optimization")
        assert len(models) == 1
        model = models[0]
        # logger.info("gradient check")
        # xx0 = np.array([model.x[cc] / model.precond[cc] for cc in model.coords])
        # f0, fp = self._f(xx0)
        # for i, cc in enumerate(model.coords):
        #     x0c = xx0.copy()
        #     x0c[i] += 1e-8
        #     f1, _ = self._f(x0c)
        #     logger.info((i, cc, f1, f0, (f1 - f0) / 1e-8, fp[i]))
        # aoeu
        res = scipy.optimize.fmin_l_bfgs_b(self._f, [model.x[cc] / model.precond[cc] for cc in model.coords], 
                None, bounds=[tuple(self._bounds[cc] / model.precond[cc]) for cc in model.coords], 
                factr=1e9)
        for xx, cc in zip(res[0], model.coords):
            model.x[cc] = xx * model.precond[cc]
        model.flatten()

class TwoPopulationOptimizer(_optimizer):
    def _f(self, xs):
        ## Craft new xs for each model based on currently defined split point
        new_x = np.zeros([2, 2, self._K])
        cc = [(a, b, c) for a, (b, c) in self._combined_coords]
        new_x.__setitem__(list(zip(*cc)), xs)
        new_x[1, self._split:] = new_x[0, self._split:]
        lls, jacs = zip(*_optimizer._f(self, new_x))
        ll = sum(lls)
        print(jacs)
        jac = []
        for a, b, c in cc:
            jac.append(jacs[a][self._coords[a].index((b, c))])
            if a == 0 and c >= self._split:
                jac[-1] += jacs[1][self._coords[1].index((b, c))]
        print((ll, jac))
        return (ll, jac)

    def _optimize(self, models):
        logger.debug("Performing a round of optimization")
        # logger.info("gradient check")
        # xx0 = np.array([model.x[cc] / model.precond[cc] for cc in model.coords])
        # f0, fp = self._f(xx0)
        # for i, cc in enumerate(model.coords):
        #     x0c = xx0.copy()
        #     x0c[i] += 1e-8
        #     f1, _ = self._f(x0c)
        #     logger.info((i, cc, f1, f0, (f1 - f0) / 1e-8, fp[i]))
        # aoeu
        x0 = [models[i].x[cc] / models[i].precond[cc] for i, cc in self._combined_coords]
        bounds = [self._bounds[cc] / models[i].precond[cc] for i, cc in self._combined_coords]
        res = scipy.optimize.fmin_l_bfgs_b(self._f, x0, None, bounds=bounds, factr=1e9) 
        self._combine_coords(models, res[0])
        for m in models:
            m.flatten()

    def _combine_coords(self, models, xx):
        for xx, (i, cc) in zip(res, self._combined_coords):
            models[i].x[cc] = xx * models[i].precond[cc]
            if i == 0:
                models[1].x[cc] = xx * models[1].precond[cc]

    def run(self, niter):
        self._split = self._K
        self._old_aic = np.inf
        i = 1
        while True:
            self._combined_coords = [(0, c) for c in self._coords[0]] + \
                    [(1, c) for c in self._coords[1] if c[1] < self._split]
            ll = _optimizer.run(self, niter)
            logger.info("Outer iteration %d" % i)
        pass

