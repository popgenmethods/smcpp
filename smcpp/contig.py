import numpy as np


class Contig:
    def __init__(self, data, n, a, fn=None):
        self.data = data
        self.n = np.array(n)
        self.a = np.array(a)
        self.fn = fn

    @property
    def npop(self):
        return len(self.n)
