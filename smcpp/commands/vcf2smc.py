from contextlib import contextmanager
import argparse
import warnings
import gzip
import itertools as it
from logging import getLogger
import numpy as np
import sys
from pysam import VariantFile
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
        if self.last_ob is not None:
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
    parser.add_argument("-i", "--distinguished_index", type=int, default=0, 
            help="index of distinguished lineage in sample ids (default: 0)")
    parser.add_argument("--missing-cutoff", metavar="c", type=int, default=10000,
            help="treat runs of homozygosity longer than <c> base pairs as missing")
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
    vcf = VariantFile(args.vcf)
    with optional_gzip(args.out, "wt") as out:
        samples = list(vcf.header.samples)
        if dist not in samples:
            raise RuntimeError("Distinguished lineage not found in data?")
        missing = [u for u in undist if u not in samples]
        if missing:
            msg = "The following samples were not found in the data: %s. " % ", ".join(missing)
            if args.ignore_missing:
                logger.warn(msg)
            else:
                msg += "If you want to continue without these samples, use --ignore-missing."
                raise RuntimeError(msg)
        undist = [u for u in undist if u not in missing]
        nb = 2 * len(undist)

        # function to convert a VCF record to our format <span, dist gt, undist gt, # undist>
        def rec2gt(rec):
            ref = rec.alleles[0]
            if None in rec.samples[dist].alleles:
                a = -1
            else:
                a = sum(allele != ref for allele in rec.samples[dist].alleles)
            bs = [allele != ref for u in undist for allele in rec.samples[u].alleles if allele is not None]
            b = sum(bs)
            nb = len(bs)
            return [a, b, nb]

        region_iterator = vcf.fetch(args.chrom, args.start, args.end)
        snps_only = (rec for rec in region_iterator if len(rec.alleles) == 2 and set(rec.alleles) <= set("ACTG"))
        with RepeatingWriter(out) as rw:
            try: 
                rec = next(snps_only)
            except StopIteration:
                raise RuntimeError("No records found in VCF for given region")
            last_pos = rec.pos
            rw.write([1] + rec2gt(rec))
            for rec in snps_only:
                abnb = rec2gt(rec)
                span = rec.pos - last_pos - 1
                if 1 <= span <= args.missing_cutoff:
                    rw.write([span, 0, 0, nb])
                elif span > args.missing_cutoff:
                    rw.write([span, -1, 0, 0])
                rw.write([1] + abnb)
                last_pos = rec.pos
