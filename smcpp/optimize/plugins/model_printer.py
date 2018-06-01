from .optimizer_plugin import OptimizerPlugin, targets
from smcpp.logging import getLogger

logger = getLogger(__name__)


class ModelPrinter(OptimizerPlugin):
    DISABLED = True

    @targets("post M-step")
    def update(self, message, *args, **kwargs):
        logger.info("Model: %s", kwargs["model"].to_s())
