import numpy as np

from .optimizer_plugin import OptimizerPlugin, targets
from smcpp.logging import getLogger
import smcpp.estimation_tools

logger = getLogger(__name__)

class HiddenStateOccupancyPrinter(OptimizerPlugin):

    @targets("post E-step")
    def update(self, message, *args, **kwargs):
        analysis = kwargs['analysis']
        hso = self.occupancy(analysis)
        logger.debug("hidden state occupancy:\n%s",
                     np.array_str(hso, precision=2))
        perp = self.perplexity(hso) / len(hso)
        logger.debug("normalized perplexity: %f", perp)
        return
        if kwargs['i'] == 0: # perp < .85:
            self.rebalance(analysis)
            hso = self.occupancy(analysis)
            logger.debug("new hidden state occupancy:\n%s",
                         np.array_str(hso, precision=2))

    def occupancy(self, analysis):
        hso = np.sum(
                [np.sum(im.getXisums(), axis=(0, 1))
                    for im in analysis._ims.values()], axis=0)
        hso /= hso.sum()
        return hso

    def perplexity(self, p):
        return np.exp(-(p * np.log(p)).sum())

    def rebalance(self, analysis):
        m = analysis.model.distinguished_model
        im = next(iter(analysis._ims.values()))
        M = len(im.hidden_states)
        hs = smcpp.estimation_tools.balance_hidden_states(
                analysis.model.distinguished_model, M)
        logger.debug("rebalanced hidden states: %s", str(hs))
        for im in analysis._ims.values():
            im.hidden_states = hs
        analysis.E_step()

