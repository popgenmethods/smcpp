from contextlib import contextmanager
import argparse
import warnings
import gzip
import itertools as it
from logging import getLogger
import numpy as np
logger = getLogger(__name__)

from ..logging import init_logging


@contextmanager
def optional_gzip(f, mode):
    with gzip.GzipFile(f, mode) if f.endswith(".gz") else open(f, mode) as o:
        yield o


class RepeatingWriter:
    def __init__(self, f):
        self.f = f
        self.last_ob = None
        self.i = 0

    def write(self, ob):
        if self.last_ob is None:
            self.last_ob = ob
            self.last_ob[0] = 0
        if ob[1:] == self.last_ob[1:]:
            self.last_ob[0] += ob[0]
        else:
            self._write_last_ob()
            self.last_ob = ob

    def _write_last_ob(self):
        self.f.write("%d %d %d %d\n" % tuple(self.last_ob))
        self.i += 1

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        print(("Wrote %d observations" % self.i))
        self._write_last_ob()


def init_parser(parser):
    parser.add_argument("--ignore-missing", default=False, action="store_true",
                        help="ignore samples which are missing in the data")
    parser.add_argument("--missing-cutoff", metavar="c", type=int, default=10000,
                        help="treat gaps in data longer than <c> base pairs as missing")
    parser.add_argument("-i", "--distinguished_index", type=int, default=0, help="index of distinguished lineage in sample ids (default: 0)")
    parser.add_argument("-s", "--start", type=int, help="starting base pair for conversion")
    parser.add_argument("-e", "--end", type=int, help="ending base pair for conversion")
    parser.add_argument("vcf", metavar="vcf[.gz]", help="input VCF file", widget="FileChooser")
    parser.add_argument("out", metavar="out[.gz]", help="output SMC++ file", widget="FileChooser")
    parser.add_argument("chrom", help="chromosome to parse")
    parser.add_argument("sample_ids", nargs="+", help="Columns to pull from the VCF, or file(s) containing the same.")


def main(args):
    init_logging(".", False)
    if len(args.sample_ids) == 1:
        try:
            with open(args.sample_ids[0], "rt") as f:
                args.sample_ids = [line.strip() for line in f]
        except IOError:
            pass
    dist = args.sample_ids[args.distinguished_index]
    undist = [sid for j, sid in enumerate(args.sample_ids) if j != args.distinguished_index]
    logger.info("Distinguished sample: " + dist)
    logger.info("Undistinguished samples: " + ",".join(undist))
    with optional_gzip(args.vcf, "rt") as vcf, optional_gzip(args.out, "wt") as out:
        strip_and_ignore = (line.rstrip() for line in vcf if line[:2] != "##")
        header = next(strip_and_ignore).strip().split()
        dist_ind = header.index(dist)
        undist_ind = []
        missing = []
        for u in undist:
            try:
                undist_ind.append(header.index(u))
            except ValueError:
                missing.append(u)
        if missing:
            msg = "The following samples were not found in the data: %s. " % ", ".join(missing)
            if args.ignore_missing:
                logger.warn(msg)
            else:
                msg += "If you want to continue without these samples, use --ignore-missing."
                raise RuntimeError(msg)
        nb = 2 * len(undist_ind)

        # function to convert a VCF row to our format <span, dist gt, undist gt, # undist>
        def row2gt(row):
            if "." in row[dist_ind][:3:2]:
                a = -1
            else:
                a = sum(int(x) for x in row[dist_ind][:3:2])
            bs = [int(x) for u in undist_ind for x in row[u][:3:2] if x != "."]
            b = sum(bs)
            nb = len(bs)
            return [a, b, nb]

        if header[0] != "#CHROM":
            raise RuntimeError("VCF file doesn't seem to have a valid header")
        chrom_filter = (line for line in strip_and_ignore if line.startswith(args.chrom))
        splitted = (line.split() for line in chrom_filter)
        start = args.start if args.start else 0
        end = args.end if args.end else np.inf
        in_region = (tup for tup in splitted if start <= int(tup[1]) <= end)
        snps_only = (tup for tup in in_region if 
                     all([x in "ACTG" for x in tup[3:5]]) and tup[6] == "PASS")
        with RepeatingWriter(out) as rw:
            tup = next(snps_only)
            last_pos = int(tup[1])
            rw.write([1] + row2gt(tup))
            for tup in snps_only:
                try:
                    abnb = row2gt(tup)
                except ValueError as e:
                    logger.warn("Error %s when attempting to parse:\n%s" % (e.message, str(tup)))
                    raise
                pos = int(tup[1])
                span = pos - last_pos - 1
                if 1 <= span <= args.missing_cutoff:
                    rw.write([span, 0, 0, nb])
                elif span > args.missing_cutoff:
                    rw.write([span, -1, 0, 0])
                rw.write([1] + abnb)
                last_pos = pos
