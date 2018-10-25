import os.path

from .optimizer_plugin import OptimizerPlugin, targets
from smcpp.logging import getLogger

logger = getLogger(__name__)


class AnalysisSaver(OptimizerPlugin):

    def __init__(self, outdir, base):
        self._outdir = outdir
        self._base = base

    def update(self, message, *args, **kwargs):
        dump = kwargs["analysis"].dump
        if message == "post E-step":
            i = kwargs["i"]
            dump(os.path.join(self._outdir, ".{}.iter{}".format(self._base, i)))
        elif message == "optimization finished":
            dump(os.path.join(self._outdir, "{}.final".format(self._base)))
