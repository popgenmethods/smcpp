import scipy.optimize

from .optimizer_plugin import OptimizerPlugin, targets
from smcpp.logging import getLogger

logger = getLogger(__name__)

class ParameterOptimizer(OptimizerPlugin):
    def __init__(self, param, bounds, target="analysis"):
        self._param = param
        self._bounds = bounds
        self._target = target

    @targets("pre M-step", no_first=False)
    def update(self, message, *args, **kwargs):
        param = self._param
        logger.info("Updating %s, bounds (%f, %f)", param, *self._bounds)
        tgt = kwargs[self._target]
        analysis = kwargs['analysis']
        if param not in ("theta", "rho", "split", "alpha"):
            raise RuntimeError("unrecognized param")
        x0 = getattr(tgt, param)
        logger.debug("Old %s: Q(%f)=%f", param, x0,
                     self._f(x0, analysis, tgt, param))
        res = scipy.optimize.minimize_scalar(self._f,
                                             args=(analysis, tgt, param),
                                             method='bounded',
                                             bounds=self._bounds)
        logger.info("New %s: %g", param, res.x)
        setattr(tgt, param, res.x)

    def _f(self, x, analysis, tgt, param):
        setattr(tgt, param, x)
        # derivatives curretly not supported for 1D optimization. not
        # clear if they really help.
        ret = -float(analysis.Q())
        logger.debug("%s f(%f)=%f", param, x, ret)
        return ret
