import numpy as np


class Contig:
    def __init__(self, data, n, a):
        self.data = data
        self.n = np.array(n)
        self.a = np.array(a)

    @property
    def npop(self):
        return len(self.n)
