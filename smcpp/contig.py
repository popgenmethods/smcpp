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

    def _windowed(self, window_size, n2=False):
        ws = window_size
        d = self.data
        span = d[:, 0]
        a = d[:, 1::3]
        b = d[:, 2::3]
        nb = d[:, 3::3]
        cs = np.cumsum(d[:, 0])
        breaks = np.arange(0, cs[-1] + ws, ws)
        bins = np.digitize(cs, breaks, right=True)
        nmiss = np.any(a >= 0, axis=1)
        muts = np.maximum(a, 0).sum(axis=1) % 2
        if not n2:
            muts += b.sum(axis=1)
            nmiss |= np.any(nb > 0, axis=1)
        muts = np.minimum(muts, 1)
        mut_counts = np.bincount(bins, weights=muts * span)
        nmiss_counts = np.bincount(bins, weights=nmiss * span)
        return bins, mut_counts, nmiss_counts

        
    def windowed_mutation_counts(self, window_size, n2=False):
        return self._windowed(window_size, n2)


    def filter_windows(self, window_size, bins, counts, lower, upper):
        d = self.data
        dropped = (lower >= counts) | (counts >= upper)
        d[dropped[bins], 1::3] = -1
        d[dropped[bins], 2::3] = 0
        d[dropped[bins], 3::3] = 0


    def __len__(self):
        return self.data[:, 0].sum()
