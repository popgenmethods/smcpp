import numpy

from .optimizer_plugin import OptimizerPlugin, targets
from logging import getLogger

logger = getLogger(__name__)


class KnotOptimizer:  # (OptimizerPlugin):
    DISABLED = True

    @targets("pre M-step")
    def update(self, message, *args, **kwargs):
        # pick a random knot and optimize
        model = kwargs["model"]
        knots = model._knots
        inds = np.arange(1, len(knots) - 1)  # hold first and last knots fixed
        for i in np.random.choice(inds, size=int(len(inds) * .2), replace=False):
            if i == 0:
                bounds = (1e-6, 0.9 * knots[1])
            elif i == len(knots) - 1:
                bounds = (1.1 * knots[-2], 2 * knots[-1])
            else:
                bounds = (knots[i - 1] * 1.1, knots[i + 1] * 0.9)
            analysis = kwargs["analysis"]
            opt = kwargs["optimizer"]
            bounds = (bounds, opt._bounds([i])[0])
            x0 = (knots[i], model[i])
            logger.info(
                "Old knot %d=(%f,%f) Q=%f", i, x0[0], x0[1], self._f(x0, analysis, i)
            )
            logger.debug("Bounds: %s", bounds)
            res = scipy.optimize.minimize(
                self._f, x0=x0, args=(analysis, i), bounds=bounds
            )
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
