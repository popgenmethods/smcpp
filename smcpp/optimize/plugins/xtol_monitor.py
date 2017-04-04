from smcpp.optimize.exceptions import EMTerminationException
from .optimizer_plugin import *
from smcpp.logging import getLogger

logger = getLogger(__name__)

class XtolMonitor(OptimizerPlugin):
    DISABLED = True
    @targets(['pre M-step', 'post M-step'], on_first=False)
    def update(self, message, *args, **kwargs):
        if message == "pre M-step":
            self._xi = self._analysis.model[:].astype('float').copy()
            return
        elif message == "post M-step":
            xp = self._analysis.model[:].astype('float').copy()
            delta = max(abs(xp - self._xi))
            logger.debug("max_i |x_i' - x_i| = %f", delta)
            if delta < self._xtol:
                logger.debug(
                        "Terminating because  ^^^^ < %f (= xtol)", 
                        self._xtol)
                raise EMTerminationException
