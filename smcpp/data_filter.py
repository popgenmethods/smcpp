import attr
import functools
import numpy as np
import sys
import wrapt
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import multiprocessing.dummy
from collections import OrderedDict
import contextlib

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
        self._results = self._files
        for f in self._filters.values():
            self._results = f(self._results)
        return self._results

    def results(self):
        yield from iter(self.run())

@attr.s
class Filter:
    def __call__(self, contigs):
        logger.debug(self)
        return self.run(contigs)

@contextlib.contextmanager
def DummyPool(*args):
    def f():
        pass
    f.map = map
    yield f

@attr.s
class ParallelFilter:
    Pool = DummyPool
    def __call__(self, contigs):
        logger.debug(self)
        with self.Pool() as p:
            return list(p.map(self.run, contigs))

@attr.s
class ProcessParallelFilter(ParallelFilter):
    Pool = multiprocessing.Pool

@attr.s
class ThreadParallelFilter(ParallelFilter):
    Pool = ThreadPoolExecutor

@attr.s
class LoadData(Filter):
    def run(self, files):
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
            assert np.max(c.a) <= 2
            assert np.min(c.a) >= 0
            assert np.sum(c.a) == 2
            assert c.data.shape[1] == 1 + 3 * len(c.n)
            logger.debug(c)
        logger.info(
                "%d population%s", len(self.populations), 
                "" if len(self.populations) == 1 else "s")
        return contigs


@attr.s
class Validate(ProcessParallelFilter):
    def run(self, c):
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
class Thin(ThreadParallelFilter):
    thinning = attr.ib(default=None)
    def run(self, c):
        thinning = self.thinning
        if thinning is None:
            thinning = (1000 * np.log(2 + c.n[0])).astype("int")   # 500  * ns
        if thinning > 1:
            logger.debug("Thinning interval: %d", thinning)
            new_data = estimation_tools.thin_data(c.data, thinning)
            c.data = new_data
        return c

@attr.s
class BinObservations(ThreadParallelFilter):
    w = attr.ib()

    def run(self, c):
        new_data = estimation_tools.bin_observations(c, self.w)
        c.data = new_data
        return c

@attr.s
class Realign(ThreadParallelFilter):
    w = attr.ib()

    def run(self, c):
        real = estimation_tools.realign(c.data, self.w)
        c.data = real
        return c

@attr.s
class CountMutations(Filter):
    w = attr.ib()

    def run(self, contigs):
        import scipy.stats.mstats
        with ThreadPoolExecutor() as pool:
            bincounts = pool.map(
                    estimation_tools.windowed_mutation_counts, 
                    contigs,
                    (self.w for _ in iter(int, 1)))
            mc = np.array([m for nmiss, muts in bincounts 
                    for m, nm in zip(muts, nmiss)
                    if nm > .5 * self.w])
        res = scipy.stats.mstats.mquantiles(mc, [0, .05, .95, 1])
        logger.debug("mutation counts in %dbp windows: min=%d .05=%d .95=%d max=%d", self.w, *res)
        self.counts = mc
        return Compress()(contigs)

@attr.s
class RecodeNonseg(Filter):
    cutoff = attr.ib()

    def run(self, contigs):
        return [estimation_tools.recode_nonseg(c, self.cutoff)
                for c in contigs]


@attr.s
class Compress(ProcessParallelFilter):
    def run(self, c):
        c.data = estimation_tools.compress_repeated_obs(c.data)
        return c


@attr.s
class BreakLongSpans(Filter):
    cutoff = attr.ib()

    def run(self, contigs):
        return [cc for c in contigs
                for cc in estimation_tools.break_long_spans(c, self.cutoff)]


@attr.s
class DropSmallContigs(Filter):
    cutoff = attr.ib()

    def run(self, contigs):
        ret = [c for c in contigs if len(c) > self.cutoff]
        if len(ret) == 0:
            logger.error("All contigs are <.01cM (estimated). Please double check your data")
            raise RuntimeError()
        return ret

@attr.s
class Watterson(Filter):
    def run(self, contigs):
        num = denom = 0
        for S, sample_sizes, spans in map(self._helper, contigs):
            num += S
            non_missing = sample_sizes > 0
            ss = sample_sizes[non_missing]
            sp = spans[non_missing]
            denom += (sp * (np.log(ss) + 0.5 / ss + 0.57721)).sum()
        self.theta_hat = num / denom
        logger.debug("sites: %d/%d\twatterson:%f", num, denom, self.theta_hat)
        return contigs

    def _helper(self, c):
        shp = [x + 1 for na in zip(c.a, c.n) for x in na]
        ret = np.zeros(shp, dtype=int)
        spans = c.data[:, 0]
        seg = (np.any(c.data[:, 1::3] >= 1, axis=1) |
               np.any(c.data[:, 2::3] >  0, axis=1))
        S = spans[seg].sum()
        sample_sizes = (c.data[:, 3::3].sum(axis=1) +
                        (c.data[:, 1::3] >= 0).sum(axis=1))
        return (S, sample_sizes, spans)


class RecodeMonomorphic(Filter):
    def run(self, contigs):
        return [self._recode(c) for c in contigs]

    def _recode(self, c):
        w = (
                np.all(c.data[:, 1::3] == c.a, axis=1) & 
                np.all(c.data[:, 2::3] == c.data[:, 3::3], axis=1)
            )
        c.data[w, 1::3] = c.data[w, 2::3] = 0
        return c
