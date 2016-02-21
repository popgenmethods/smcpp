import numpy as np
import scipy.optimize
import logging
logger = logging.getLogger(__name__)

class _optimizer(object):
    def __init__(self, iserv, bounds, penalty):
        self._iserv = iserv
        self._bounds = bounds
        self._penalty = penalty
        self._coords = iserv.coords
        self._precond = iserv.precond

    def _f(self, xs):
        models = self._iserv.model
        regs = []
        for m, xx, coords, precond in zip(models, xs, self._coords, self._precond):
        # Preconditioner (sort of)
            for xxx, cc in zip(xx, coords):
                m.x[cc] = xxx * precond[cc]
            m.flatten()
            regs.append(m.regularizer(self._penalty))
        self._iserv.set_params([(m, c) for m, c in zip(models, self._coords)])
        lls, jacs = zip(*[zip(*r) for r in self._iserv.Q()])
        ret = [-np.mean(lls, axis=0), -np.mean(jacs, axis=0)]
        reg, dreg = regularizer([aa, bb], coords, ctx.lambda_penalty)
        ret[0] += reg
        ret[1] += dreg
        dary = np.zeros([2, ctx.model.K])
        for i, cc in enumerate(coords):
            ret[1][i] *= ctx.precond[cc]
            dary[cc] = ret[1][i]
        logger.debug(x0c)
        logger.debug(dary)
        logger.debug(ret[0])
        logger.debug("regularizer: %s" % str((reg, dreg)))
        return ret

class SinglePopulationOptimizer(_optimizer):
    def _f(self, xs):
        return _optimizer._f(self, [xs])

    def _optimize(self, model):
        logger.debug("Performing a round of optimization")
        res = scipy.optimize.fmin_l_bfgs_b(self._f, [model.x[cc] / model.precond[cc] for cc in model.coords], 
                None, bounds=[tuple(self._bounds[cc] / model.precond[cc]) for cc in model.coords], 
                factr=1e9)
        ret = np.array([x * model.precond[cc] for x, cc in zip(res[0], model.coords)])
        return ret

    def run(self, niter):
        logger.debug("Initializing model(s)")
        model = self._iserv.model[0]
        self._iserv.set_params([[model, False]])
        logger.debug("Performing initial E step")
        self._iserv.E_step(False)
        llold = sum(self._iserv.loglik()[0])
        logger.debug("Starting loglik: %g" % llold)
        for i in range(niter):
            logger.info("EM iteration %d/%d" % (i + 1, niter))
            logger.info("\tM-step...")
            ret = self._optimize(model)
            for xx, cc in zip(ret, model.coords):
                model.x[cc] = xx
            model.flatten()
            logger.debug("Current model:\n%s", str(ctx.model.x))
            iserv.set_params([(model, False)])
            logger.info("\tE-step...")
            iserv.E_step()
            ll = np.sum(iserv.loglik())
            if i > 0:
                logger.info("\tNew/old loglik: %f/%f" % (ll, llold))
            if ll < llold:
                logger.warn("Log-likelihood decreased")
            llold = ll
            pop0 = iserv.populations[0]
            esfs = _smcpp.sfs(n, model.x, 0.0, _smcpp.T_MAX, pop0.theta, False)
            model.dump(os.path.join(args.outdir, ".model.%d" % i))
            logger.debug("model sfs:\n%s" % str(esfs))
            logger.debug("observed sfs:\n%s" % str(pop0.obsfs))
            i += 1
        model.dump(os.path.join(args.outdir, "model.final" % i))

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

