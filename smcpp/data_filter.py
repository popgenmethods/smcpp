from dataclasses import dataclass, field
from typing import Sequence, List
import functools
import numpy as np
import sys
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
import contextlib
import multiprocessing


from . import estimation_tools, defaults

import logging
logger = logging.getLogger(__name__)
mp_ctx = multiprocessing.get_context("forkserver")

@dataclass
class Filter:

    def __call__(self, contigs):
        logger.debug(self)
        return self.run(contigs)



@dataclass
class DataPipeline:
    files: Sequence[str]
    _filters: OrderedDict = field(init=None, default_factory=OrderedDict)
    _results: List = None

    def __getitem__(self, key):
        self.run()
        return self._filters[key]

    def add_filter(self, *args, **kwargs):
        """
        Add filter to pipeline. Filters are executed "inside out" (i.e. in
        reverse order.
        """
        assert (len(args) == 0) != (len(kwargs) == 0)
        if kwargs:
            self._filters.update(kwargs)
        else:
            self._filters["filter%d" % len(self._filters)] = args[0]
        self._results = None

    def run(self):
        if self._results is not None:
            return self._results
        self._results = self.files
        for f in self._filters.values():
            self._results = f(self._results)
        return self._results

    def results(self):
        yield from iter(self.run())


@contextlib.contextmanager
def DummyPool(*args):

    def f():
        pass

    f.map = map
    yield f


@dataclass
class ParallelFilter:
    Pool = DummyPool

    def __call__(self, contigs):
        logger.debug(self)
        with self.Pool() as p:
            return list(p.map(self.run, contigs))


@dataclass
class ProcessParallelFilter(ParallelFilter):
    Pool = mp_ctx.Pool


@dataclass
class ThreadParallelFilter(ParallelFilter):
    Pool = ThreadPoolExecutor


@dataclass
class LoadData(Filter):

    def run(self, files):
        ## Parse each data set into an array of observations
        logger.info("Loading data...")
        files = estimation_tools.files_from_command_line_args(files)
        contigs = estimation_tools.load_data(files)
        L = sum(c.data[:, 0].sum() for c in contigs)
        logger.info("%.2f Gb of data", L * 1e-9)
        pops = set(c.pid for c in contigs)
        unique_pops = list({x for p in pops for x in p})
        assert len(unique_pops) <= 2, (
            "Only one or two populations are supported, but the "
            "following were found in the data: %r" % unique_pops
        )
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
            "%d population%s",
            len(self.populations),
            "" if len(self.populations) == 1 else "s",
        )
        return contigs


@dataclass
class Validate(ProcessParallelFilter):

    def run(self, c):
        assert c.data.flags.c_contiguous
        nonseg = (
            (
                np.all(c.data[:, 1::3] == c.a[None, :], axis=1)
                | np.all(c.data[:, 1::3] == -1, axis=1)
            )
            & np.all(c.data[:, 2::3] == c.data[:, 3::3], axis=1)
            & np.any(c.data[:, 3::3] > 0, axis=1)
        )
        if np.any(nonseg):
            logger.debug("In file %s, observations %s:", c.fn, np.where(nonseg)[0])
            logger.debug(
                "Data set contains sites where every "
                "individual is homozygous for the derived allele."
            )
            a = c.data[nonseg, 1::3]
            a[a >= 0] = 0
            c.data[nonseg, 2::3] = 0
        bad = c.data[:, 0] <= 0 | np.any(
            c.data[:, 1::3] > c.a[None, :], axis=1
        ) | np.any(c.data[:, 2::3] > c.data[:, 3::3], axis=1) | np.any(
            c.data[:, 3::3] > c.n[None, :], axis=1
        )
        if np.any(bad):
            logger.error(
                "File %s has invalid observations "
                "(span <= 0 | a > 2 | b > n | n > sample size): %s",
                c.fn,
                np.where(bad)[0],
            )
            raise RuntimeError("data validation failed")
        return c


@dataclass
class Thin(ThreadParallelFilter):
    thinning: int = None

    def run(self, c):
        thinning = self.thinning
        if thinning is None:
            thinning = (500 * np.log(2 + c.n[0])).astype("int")  # 500  * ns
        if thinning > 1:
            logger.debug("Thinning interval: %d", thinning)
            new_data = estimation_tools.thin_data(c.data, thinning)
            c.data = new_data
        return c


@dataclass
class BinObservations(ThreadParallelFilter):
    w: int

    def run(self, c):
        new_data = estimation_tools.bin_observations(c, self.w)
        c.data = new_data
        return c


@dataclass
class Realign(ThreadParallelFilter):
    w: int

    def run(self, c):
        real = estimation_tools.realign(c.data, self.w)
        c.data = real
        return c


@dataclass
class Chunk(ThreadParallelFilter):
    w: int

    def run(self, c):
        d = estimation_tools.realign(c.data, self.w)
        inds = np.where(np.cumsum(d[:, 0]) % self.w == 0)[0]
        return [x for x in np.split(d, 1 + inds) if x[:, 0].sum() == self.w]


@dataclass
class CountMutations(Filter):
    w: int

    def run(self, contigs):
        import scipy.stats.mstats

        with ThreadPoolExecutor() as pool:
            bincounts = pool.map(
                estimation_tools.windowed_mutation_counts,
                contigs,
                (self.w for _ in iter(int, 1)),
            )
            mc = np.array(
                [
                    m * self.w / nm
                    for nmiss, muts in bincounts
                    for m, nm in zip(muts, nmiss)
                    if nm > .5 * self.w
                ]
            )
        res = scipy.stats.mstats.mquantiles(mc, [0, .05, .95, 1])
        logger.debug(
            "mutation counts in %dbp windows: min=%d .05=%d .95=%d max=%d", self.w, *res
        )
        self.counts = mc
        return contigs


@dataclass
class RecodeNonseg(Filter):
    cutoff: int

    def run(self, contigs):
        return [estimation_tools.recode_nonseg(c, self.cutoff) for c in contigs]


@dataclass
class Compress(ProcessParallelFilter):

    def run(self, c):
        c.data = estimation_tools.compress_repeated_obs(c.data)
        return c


@dataclass
class BreakLongSpans(Filter):
    cutoff: int

    def run(self, contigs):
        return [
            cc
            for c in contigs
            for cc in estimation_tools.break_long_spans(c, self.cutoff)
        ]


@dataclass
class DropUninformativeContigs(Filter):

    def _n_variable_sites(self, c):
        d = c.data
        return ((d[:, 1::3].sum(axis=1) > 0) | (d[:, 2::3].sum(axis=1) > 0)).sum()

    def run(self, contigs):
        ret = []
        for c in contigs:
            if self._n_variable_sites(c) > 0:
                ret.append(c)
            else:
                logger.debug(
                    "Dropping a contig derived from %s which has no mutations.", c.fn
                )
        if len(ret) == 0:
            logger.error("No contigs have mutation data. Inference is impossible.")
            raise RuntimeError()
        return ret


@dataclass
class DropSmallContigs(Filter):
    cutoff: int

    def run(self, contigs):
        ret = [c for c in contigs if len(c) > self.cutoff]
        if len(ret) == 0:
            logger.error(
                "All contigs are <.01cM (estimated). " "Please double check your data."
            )
            raise RuntimeError()
        return ret


@dataclass
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
        seg = np.any(c.data[:, 1::3] >= 1, axis=1) | np.any(c.data[:, 2::3] > 0, axis=1)
        S = spans[seg].sum()
        sample_sizes = c.data[:, 3::3].sum(axis=1) + (c.data[:, 1::3] >= 0).sum(axis=1)
        return (S, sample_sizes, spans)


@dataclass
class RecodeMonomorphic(Filter):

    def run(self, contigs):
        return [self._recode(c) for c in contigs]

    def _recode(self, c):
        w = np.all(c.data[:, 1::3] == c.a, axis=1) & np.all(
            c.data[:, 2::3] == c.data[:, 3::3], axis=1
        )
        c.data[w, 1::3] = c.data[w, 2::3] = 0
        return c


@dataclass
class Summarize(Filter):

    def run(self, contigs):
        for c in contigs:
            logger.debug(c.data[:10])
        return contigs
