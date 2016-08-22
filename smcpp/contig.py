class Contig:
    def __init__(self, data, n, a):
        self.data = data
        self.n = n
        self.a = a

    @property
    def npop(self):
        return len(self.n)
