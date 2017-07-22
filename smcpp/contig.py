import numpy as np
import attr


@attr.s
class Contig:
    pid = attr.ib()
    data = attr.ib()
    n = attr.ib(convert=np.array)
    a = attr.ib(convert=np.array)
    fn = attr.ib(default=None)

    @property
    def npop(self):
        return len(self.pid)

    @property
    def key(self):
        return (self.pid, tuple(self.n), tuple(self.a))
