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

    def _f(self, xs):
        models = self._iserv.model
        regs = []
        for m, xx, coords, precond in zip(models, xs, self._coords, self._precond):
        # Preconditioner (sort of)
            for xxx, cc in zip(xx, coords):
                m.x[cc] = xxx * precond[cc]
            m.flatten()
            regs.append(m.regularizer(self._cmdargs.lambda_penalty))
        self._iserv.set_params(models, self._coords)
        ret = []
        for q, reg, coords, precond in zip(self._iserv.Q(), regs, self._coords, self._precond):
            lls, jacs = zip(*q)
            tmp = [-np.mean(lls), -np.mean(jacs, axis=0)]
            tmp[0] += reg[0]
            tmp[1] += reg[1]
            for i, cc in enumerate(coords):
                tmp[1][i] *= precond[cc]
            ret.append(tmp)
        return list(map(sum, zip(*ret)))

class SinglePopulationOptimizer(_optimizer):
    def _f(self, xs):
        ret = _optimizer._f(self, [xs])
        return ret

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

class TwoPopulationOptimizer(object):
    def __init__(self, iserv):
        self._iserv = iserv
    def run(self):
        pass

#     ## Optimization stuff 
#     i = 0
#     coords = [(aa, j) for j in range(ctx.model.K) for aa in ((0,) if j in ctx.flat_pieces else (0, 1))]
#     # Vector of "preconditioners" helps with optimization
#     ctx.precond = {coord: 1. / ctx.model.s[coord[1]] for coord in coords}
#     # ctx.precond = {coord: 1. for coord in coords}
#     if (ctx.model.K - 1, 0) in coords:
#         ctx.precond[(ctx.model.K - 1, 0)] = 1. / (15.0 - np.sum(ctx.model.s))
#     while i < args.em_iterations:
#         logger.info("EM iteration %d/%d" % (i + 1, args.em_iterations))
#         logger.info("\tM-step...")
#         ret = optimize(coords, args.lbfgs_factor)
#         for xx, cc in zip(ret, coords):
#             ctx.model.x[cc] = xx
#         ctx.model.b[ctx.flat_pieces] = ctx.model.a[ctx.flat_pieces]
#         logger.debug("Current model:\n%s", str(ctx.model.x))
#         ctx.im.set_params(ctx.model.x, False)
#         logger.info("\tE-step...")
#         ctx.im.E_step()
#         ll = np.sum(ctx.im.loglik())
#         if i > 0:
#             logger.info("\tNew/old loglik: %f/%f" % (ll, ctx.llold))
#         if ll < ctx.llold:
#             logger.warn("Log-likelihood decreased")
#         ctx.llold = ll
#         esfs = _smcpp.sfs(n, ctx.model.x, 0.0, ctx.model.hidden_states[-1], ctx.model.theta, False)
#         write_model(os.path.join(args.outdir, "model.%d.txt" % i))
#         logger.debug("model sfs:\n%s" % str(esfs))
#         logger.debug("observed sfs:\n%s" % str(obsfs))
#         i += 1
#     write_model(os.path.join(args.outdir, "model.final.txt"))

# def optimize(coords, factr):
#     def fprime(x):
#         x0c = ctx.model.x.copy()
#         # Preconditioner (sort of)
#         for xx, cc in zip(x, coords):
#             x0c[cc] = xx * ctx.precond[cc]
#         aa, bb, _ = x0c
#         bb[ctx.flat_pieces] = aa[ctx.flat_pieces]
#         ctx.im.set_params((aa, bb, ctx.model.s), coords)
#         res = ctx.im.Q()
#         lls = np.array([ll for ll, jac in res])
#         jacs = np.array([jac for ll, jac in res])
#         ret = [-np.mean(lls, axis=0), -np.mean(jacs, axis=0)]
#         reg, dreg = regularizer([aa, bb], coords, ctx.lambda_penalty)
#         ret[0] += reg
#         ret[1] += dreg
#         dary = np.zeros([2, ctx.model.K])
#         for i, cc in enumerate(coords):
#             ret[1][i] *= ctx.precond[cc]
#             dary[cc] = ret[1][i]
#         logger.debug(x0c)
#         logger.debug(dary)
#         logger.debug(ret[0])
#         logger.debug("regularizer: %s" % str((reg, dreg)))
#         return ret
#     # logger.debug("gradient check")
#     # xx0 = np.array([ctx.x[cc] / ctx.precond[cc] for cc in coords])
#     # f0, fp = fprime(xx0)
#     # for i, cc in enumerate(coords):
#     #     x0c = xx0.copy()
#     #     x0c[i] += 1e-8
#     #     f1, _ = fprime(x0c)
#     #     logger.debug((i, cc, f1, f0, (f1 - f0) / 1e-8, fp[i]))
