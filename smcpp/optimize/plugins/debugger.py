import os
import numpy as np

from .optimizer_plugin import OptimizerPlugin, targets
from smcpp.logging import getLogger

logger = getLogger(__name__)

class Debugger(OptimizerPlugin):
    DISABLED = True
    @targets("post minimize")
    def update(self, message, *args, **kwargs):
        if os.environ.get("SMCPP_DEBUG"):
            y = input("Break? ")
            if len(y) and y[0] == "y":
                im = next(iter(kwargs['analysis']._ims.values()))
                gs = im.gamma_sums
                z = np.zeros_like(list(gs[0].values())[0])
                d = {}
                for g in gs:
                    for k in g:
                        d.setdefault(k, z.copy())
                        d[k] += g[k]
                gs = d
                xis = np.sum(im.xisums, axis=0)
                import ipdb
                ipdb.set_trace()
