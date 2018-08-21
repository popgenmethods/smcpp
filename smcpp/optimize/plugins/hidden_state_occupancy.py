import numpy as np

from .optimizer_plugin import OptimizerPlugin, targets
from smcpp.logging import getLogger
import smcpp.estimation_tools

logger = getLogger(__name__)


class HiddenStateOccupancyPrinter(OptimizerPlugin):

    def __init__(self, *args, **kwargs):
        super(*args, **kwargs)
        self._last_perp = None

    @targets(["pre E-step", "post E-step"])
    def update(self, message, *args, **kwargs):
        analysis = kwargs["analysis"]
        if kwargs["i"] == 0:
            return
        hso = self.occupancy(analysis)
        logger.debug("hidden state occupancy:\n%s", np.array_str(hso, precision=2))
        perp = self.perplexity(hso) / len(hso)
        logger.debug("normalized perplexity: %f", perp)
        return
        if message == "post E-step" and perp < smcpp.defaults.perplexity_threshold:
            self._last_perp = perp
            self.rebalance(analysis)
            analysis.E_step()

    def occupancy(self, analysis):
        hso = np.sum(
            [np.sum(im.xisums, axis=(0, 1)) for im in analysis._ims.values()], axis=0
        )
        hso /= hso.sum()
        return hso

    def perplexity(self, p):
        return np.exp(-(p * np.log(p)).sum())

    def rebalance(self, analysis):
        m = analysis.model.distinguished_model
        im = next(iter(analysis._ims.values()))
        self._last_hs = im.hidden_states.copy()
        M = len(self._last_hs) - len(m.knots)
        hs = analysis.rescale(
            smcpp.estimation_tools.balance_hidden_states(
                analysis.model.distinguished_model, M
            )
        )
        hs = np.sort(np.r_[hs, m.knots])
        logger.debug("rebalanced hidden states: %s", np.array_str(hs, precision=2))
        for im in analysis._ims.values():
            im.hidden_states = hs
