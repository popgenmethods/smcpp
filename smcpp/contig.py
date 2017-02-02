import numpy as np


class Contig:
    def __init__(self, pid, data, n, a, fn=None):
        self.pid = pid
        self.data = data
        self.n = np.array(n)
        self.a = np.array(a)
        self.fn = fn

    @property
    def npop(self):
        return len(self.pid)

    @property
    def key(self):
        return (self.pid, tuple(self.n), tuple(self.a))
