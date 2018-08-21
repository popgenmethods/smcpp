import numpy as np

from .optimizer_plugin import OptimizerPlugin, targets
from smcpp.logging import getLogger

logger = getLogger(__name__)


class TransitionDebug:  # (OptimizerPlugin):

    @classmethod
    def set_path(cls, path):
        cls._path = path

    def __init__(self):
        try:
            os.makedirs(self._path)
        except OSError:
            pass

    @targets("post minimize")
    def update(self, message, *args, **kwargs):
        k = next(iter(kwargs["analysis"]._ims))
        im = kwargs["analysis"]._ims[k]
        T = im.transition
        xis = np.sum(im.xisums, axis=0)
        np.savetxt(os.path.join(self._path, "xis.txt"), xis.astype("float"), fmt="%g")
        log_T = np.array(ad.admath.log(T))
        np.savetxt(
            os.path.join(self._path, "log_T.txt"), log_T.astype("float"), fmt="%g"
        )
        q3 = log_T * xis
        np.savetxt(os.path.join(self._path, "q3.txt"), q3.astype("float"), fmt="%g")
        for i, d in enumerate(im.model[:]):
            f = np.vectorize(lambda x, d=d: x.d(d))
            np.savetxt(
                os.path.join(self._path, "log_T.%d.txt" % i),
                f(log_T).astype("float"),
                fmt="%g",
            )
            np.savetxt(
                os.path.join(self._path, "q3.%d.txt" % i),
                f(q3).astype("float"),
                fmt="%g",
            )
