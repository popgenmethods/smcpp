from .optimizer_plugin import OptimizerPlugin, targets
from logging import getLogger

logger = getLogger(__name__)


class ProgressPrinter(OptimizerPlugin):

    def update(self, message, *args, **kwargs):
        if message == "begin":
            logger.info("Starting EM algorithm...")
        if message == "pre E-step":
            logger.info("EM iteration %d of %d...", kwargs["i"] + 1, kwargs["niter"])
        if message == "M step":
            logger.debug("Optimizing coordinates %s", kwargs["coords"])
        if message == "post mini M-step":
            logger.debug("Results of optimizer:\n%s", kwargs["res"])
        if message == "post M-step":
            logger.info("Current model:\n%s", kwargs["model"].to_s())
