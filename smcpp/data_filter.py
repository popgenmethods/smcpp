import attr
import functools
import numpy as np
import sys
import wrapt
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from collections import OrderedDict

from . import logging, estimation_tools, defaults

logger = logging.getLogger(__name__)

@attr.s
class DataPipeline:
    _files = attr.ib()
    _filters = attr.ib(init=False, default=attr.Factory(OrderedDict))
    _results = attr.ib(init=False, default=None)

    def __getitem__(self, key):
        self.run()
        return self._filters[key]

    def add_filter(self, *args, **kwargs):
        '''
        Add filter to pipeline. Filters are executed "inside out" (i.e. in
        reverse order.
        '''
        assert (len(args) == 0) != (len(kwargs) == 0)
        if kwargs:
            self._filters.update(kwargs)
        else:
            self._filters['filter%d' % len(self._filters)] = args[0]

    def run(self):
        if self._results is not None:
            return self._results
        def comp(f, g):
            return lambda x: f(g(x))
        h = functools.reduce(comp, reversed(self._filters.values()), lambda x: x)
        self._results = h(self._files)
        return self._results

    def results(self):
        yield from iter(self.run())


def parallel(threaded):
    @wrapt.decorator
    def wrapper(wrapped, instance, args, kwargs):
        return list(map(wrapped, args[0]))
        Pool = ThreadPoolExecutor if threaded else ProcessPoolExecutor
        with Pool(defaults.cores) as p:
            ret = list(p.map(wrapped, args[0]))
    return wrapper

@attr.s
class Filter:
    pass

@attr.s
class LoadData(Filter):
    def __call__(self, files):
        ## Parse each data set into an array of observations
        logger.info("Loading data...")
        files = estimation_tools.files_from_command_line_args(files)
        contigs = estimation_tools.load_data(files)
        L = sum(c.data[:,0].sum() for c in contigs)
        logger.info("%.2f Gb of data", L * 1e-9)
        pops = set(c.pid for c in contigs)
        unique_pops = list({x for p in pops for x in p})
        assert len(unique_pops) <= 2, (
                "Only one or two populations are supported, but the "
                "following were found in the data: %r" % unique_pops)
        assert len(unique_pops) <= 2
        self.populations = tuple(unique_pops)
        for c in contigs:
            assert len(c.n) == len(c.a)
            assert c.a.max() <= 2
            assert c.a.min() >= 0
            assert c.a.sum() == 2
            assert c.data.shape[1] == 1 + 3 * len(c.n)
            logger.debug(c)
        logger.info(
                "%d population%s", len(self.populations), 
                "" if len(self.populations) == 1 else "s")
        return contigs


@attr.s
class Validator(Filter):

    @parallel(False)
    def __call__(self, c):
        assert c.data.flags.c_contiguous
        nonseg = ((np.all(c.data[:, 1::3] == c.a[None, :], axis=1) |
                   np.all(c.data[:, 1::3] == -1, axis=1)) &
                   np.all(c.data[:, 2::3] == c.data[:, 3::3], axis=1) &
                   np.any(c.data[:, 3::3] > 0, axis=1))
        if np.any(nonseg):
            logger.error("In file %s, observations %s:", c.fn, np.where(nonseg)[0])
            logger.error("Data set contains sites where every "
                    "individual is homozygous for the derived allele. "
                    "Please recode these as non-segregating (homozygous ancestral).")
            raise RuntimeError("data validation failed")
        bad = (c.data[:, 0] <= 0 |
               np.any(c.data[:, 1::3] > c.a[None, :], axis=1) |
               np.any(c.data[:, 2::3] > c.data[:, 3::3], axis=1) |
               np.any(c.data[:, 3::3] > c.n[None, :], axis=1))
        if np.any(bad):
            logger.error("File %s has invalid observations "
                         "(span <= 0 | a > 2 | b > n | n > sample size): %s",
                         c.fn, np.where(bad)[0])
            raise RuntimeError("data validation failed")
        return c

@attr.s
class BinObservations(Filter):
    w = attr.ib()
    thinning = attr.ib(default=None)

    def __call__(self, contigs):
        contigs = Realign(w=self.w)(contigs)
        return self._run_individual(contigs)

    @parallel(True)
    def _run_individual(self, c):
        bins, muts, nmiss = estimation_tools.windowed_mutation_counts(c.data, self.w)
        new_data = np.zeros([muts.shape[0], c.data.shape[1]], dtype=np.int32)
        assert c.npop == 1
        new_data[muts == 1, 1] = 1
        new_data[nmiss == 0, 1] = -1
        thinning = self.thinning
        if thinning is None:
            thinning = (1000 * np.log(2 + c.n[0])).astype("int")   # 500  * ns
        th = thinning // self.w
        for i in range(1, new_data.shape[0], th):
            new_data[i] = -np.sort(-c.data[bins == i, ::-1], axis=0)[0, ::-1]
        new_data[:, 0] = 1
        c.data = new_data
        return c

@attr.s
class Realign(Filter):
    w = attr.ib()

    @parallel(True)
    def __call__(self, c):
        real = estimation_tools.realign(c.data, self.w)
        c.data = real
        return c

@attr.s
class CountMutations(Filter):
    w = attr.ib()

    def __call__(self, contigs):
        import scipy.stats.mstats
        with ProcessPoolExecutor() as pool:
            bincounts = map(
                    estimation_tools.windowed_mutation_counts, 
                    (c.data for c in contigs),
                    (self.w for _ in iter(int, 1)))
            mc = [m for _, muts, nmiss in bincounts 
                    for m, nm in zip(muts, nmiss)
                    if nm > .5 * self.w]
        res = scipy.stats.mstats.mquantiles(mc, [0, .05, .95, 1])
        logger.debug("mutation counts: min=%d .05=%d .95=%d max=%d", *res)
        self.counts = mc
        return contigs

@attr.s
class RecodeNonseg(Filter):
    cutoff = attr.ib()

    @parallel(False)
    def __call__(self, c):
        return estimation_tools.recode_nonseg(c, self.cutoff)


@attr.s
class Compress(Filter):
    @parallel(False)
    def __call__(self, c):
        c.data = estimation_tools.compress_repeated_obs(c.data)
        return c


@attr.s
class BreakLongSpans(Filter):
    cutoff = attr.ib()

    @parallel(False)
    def __call__(self, c):
        return estimation_tools.break_long_spans(c, self.cutoff)


@attr.s
class Combiner(Filter):
    def __call__(self, contigs):
        return [c for x in contigs for c in x]


@attr.s
class DropSmallContigs(Filter):
    cutoff = attr.ib()

    def __call__(self, contigs):
        ret = [c for c in contigs if len(c) > self.cutoff]
        if len(ret) == 0:
            logger.error("All contigs are <.01cM (estimated). Please double check your data")
            raise RuntimeError()
        return ret
