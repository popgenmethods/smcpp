from .optimizer_plugin import OptimizerPlugin, targets
from logging import getLogger
from smcpp.optimize.exceptions import EMTerminationException

logger = getLogger(__name__)


class LoglikelihoodMonitor(OptimizerPlugin):

    def __init__(self):
        self._old_loglik = None

    @targets("post E-step")
    def update(self, message, *args, **kwargs):
        ll = kwargs["analysis"].loglik()
        if self._old_loglik is None:
            logger.info("Loglik: %f", ll)
        else:
            improvement = (self._old_loglik - ll) / self._old_loglik
            logger.info(
                "New loglik: %f\t(old: %f [%f%%])",
                ll,
                self._old_loglik,
                100. * improvement,
            )
            tol = kwargs["optimizer"]._ftol
            if improvement < 0:
                logger.warn("Loglik decreased")
            elif improvement < tol:
                logger.info("Log-likelihood improvement < tol=%g; terminating", tol)
                raise EMTerminationException()
        self._old_loglik = ll
