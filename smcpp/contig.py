import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Contig:
    pid: Tuple
    data: np.ndarray
    n: int
    a: int
    fn: str = None

    def __post_init__(self):
        self.n = np.array(self.n)
        self.a = np.array(self.a)

    @property
    def npop(self):
        return len(self.pid)

    @property
    def key(self):
        return (self.pid, tuple(self.n), tuple(self.a))

    def __len__(self):
        return self.data[:, 0].sum()
