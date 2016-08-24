import argparse
import warnings
import itertools as it
from logging import getLogger
import numpy as np
import sys
from pysam import VariantFile, TabixFile
import json
from collections import Counter
logger = getLogger(__name__)

from ..logging import setup_logging
from ..util import optional_gzip, RepeatingWriter
from ..version import __version__


def comma_separated_list(x):
    return x.split(",")


def comma_separated_fixed_list(lower, upper):
    def f(x):
        ret = comma_separated_list(x)
        if not lower <= len(ret) <= upper:
            raise argparse.ArgumentTypeError(
                    "%r is not a comma-separated list of between %d and %d elements" %
                    (x, lower, upper))
        return ret
    return f


def init_parser(parser):
    parser.add_argument("--apart", "-a", action="store_true",
            help="If specified, distinguished lineages will be pulled from "
                 "from first sample in population 1 and second sample in population 2. "
                 "This option only makes sense for phased data with two populations. "
                 "(Default: both lineages from first sample in population 1.) ")
    parser.add_argument("-i", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--ignore-missing", default=False, action="store_true",
            help="ignore samples which are missing in the data")
    parser.add_argument("--missing-cutoff", "-c", metavar="c", type=int, default=None,
            help="treat runs of homozygosity longer than <c> base pairs as missing")
    parser.add_argument("--mask", "-m",
            help="BED-formatted mask of missing regions",
            widget="FileChooser")
    parser.add_argument("vcf", metavar="vcf[.gz]",
            help="input VCF file", widget="FileChooser")
    parser.add_argument("out", metavar="out[.gz]",
            help="output SMC++ file", widget="FileChooser")
    parser.add_argument("contig", help="contig to parse")
    parser.add_argument("pop1", type=comma_separated_list,
            help="Comma-separated list of sample ids from population 1.")
    parser.add_argument("pop2", type=comma_separated_list, nargs="?", default=[],
            help="Comma-separated list of sample ids from population 2.")


def validate(args):
    if args.missing_cutoff and args.mask:
        raise RuntimeError("--missing-cutoff and --mask are mutually exclusive")


def main(args):
    setup_logging(0)
    validate(args)
    for i in [1, 2]:
        attr = "pop%d" % i
        ary = getattr(args, attr)
        if len(ary) == 1 and ary[0][0] == "@":
            setattr(args, attr, open(ary[0][1:], "rt").read().strip().split("\n"))
    for i, p in enumerate([args.pop1, args.pop2], 1):
        if p:
            c = Counter(p)
            if max(c.values()) > 1:
                raise RuntimeError(
                        "Pop %d has duplicated samples: %s" %
                        (i, [item for item in c.items() if item[1] > 1]))
    if args.apart:
        if not args.pop2:
            raise RuntimeError("--apart requires two populations")
        dist = [[args.pop1[args.i]], [args.pop2[args.i]]]
        undist = [args.pop1, args.pop2]
    else:
        dist = [[args.pop1[args.i]] * 2, []]
        undist = [args.pop1[:i] + args.pop1[args.i + 1:], args.pop2]
    npop = 1
    logger.info("Population 1:")
    logger.info("Distinguished lineages: " + 
                ", ".join(["%s:%d" % c[::-1] for c in enumerate(dist[0], 1)]))
    logger.info("Undistinguished samples: " + ", ".join(undist[0]))
    if args.pop2:
        npop = 2
        common = set(args.pop1) & set(args.pop2)
        if common:
            logger.error("Populations 1 and 2 should be disjoint, "
                         "but both contain " + ", ".join(common))
            sys.exit(1)
        logger.info("Population 2:")
        logger.info("Distinguished lineages: " +
                    ", ".join(["%s:%d" % c[::-1] for c in enumerate(dist[1], 1)]))
        logger.info("Undistinguished samples: " + ", ".join(undist[1]))

    ## Start parsing
    vcf = VariantFile(args.vcf)
    with optional_gzip(args.out, "wt") as out:
        samples = list(vcf.header.samples)
        dist = dist[:npop]
        undist = undist[:npop]
        if not all([set(d) <= set(samples) for d in dist]):
            raise RuntimeError("Distinguished lineages not found in data?")
        missing = [samp for u in undist for samp in u if samp not in samples]
        if missing:
            msg = "The following samples were not found in the data: %s. " % ", ".join(missing)
            if args.ignore_missing:
                logger.warn(msg)
            else:
                msg += "If you want to continue without these samples, use --ignore-missing."
                raise RuntimeError(msg)
        undist = [[sample for sample in u if sample not in missing] for u in undist]

        dis = [list(enumerate(d)) for d in dist]
        undis = [[(i, d) for i in range(2) for d in und if (i, d) not in di]
                 for di, und in zip(dis, undist)]
        out.write("# SMC++ ")
        json.dump({"__version__": __version__, "undist": undis, "dist": dis}, out)
        out.write("\n")
        na = [len(d) for d in dis]
        nb = [len(u) for u in undis]
        # function to convert a VCF record to our format <span, dist gt, undist gt, # undist>
        def rec2gt(rec):
            ref = rec.alleles[0]
            da = [[rec.samples[d].alleles[i] for i, d in di] for di in dis]
            a = [sum(x != ref for x in d) if None not in d else -1
                 for d in da]
            bs = [[rec.samples[d].alleles[i] != ref for i, d in undi
                   if rec.samples[d].alleles[i] is not None]
                  for undi in undis]
            b = [sum(_) for _ in bs]
            nb = [len(_) for _ in bs]
            # Fold non-polymorphic (in subsample) sites
            if np.array_equal(b, nb) and np.array_equal(a, na):
                a = [0] * len(a)
                b = [0] * len(b)
            return list(sum(zip(a, b, nb), tuple()))

        region_iterator = vcf.fetch(contig=args.contig)
        contig_length = vcf.header.contigs[args.contig].length
        if args.mask:
            mask_iterator = TabixFile(args.mask).fetch(reference=args.contig)
            args.missing_cutoff = np.inf
        else:
            mask_iterator = iter([])
            if args.missing_cutoff is None:
                args.missing_cutoff = np.inf
        mask_iterator = (x.split("\t") for x in mask_iterator)
        mask_iterator = ((x[0], int(x[1]), int(x[2])) for x in mask_iterator)
        snps_only = (rec for rec in region_iterator if len(rec.alleles) == 2 and set(rec.alleles) <= set("ACTG"))

        def interleaved():
            cmask = next(mask_iterator, None)
            csnp = next(snps_only, None)
            while cmask or csnp:
                if cmask is None:
                    yield "snp", csnp
                    csnp = next(snps_only, None)
                elif csnp is None:
                    yield "mask", cmask
                    cmask = next(mask_iterator, None)
                else:
                    if csnp.pos < cmask[1]:
                        yield "snp", csnp
                        csnp = next(snps_only, None)
                    elif csnp.pos <= cmask[2]:
                        while csnp is not None and csnp.pos <= cmask[2]:
                            csnp = next(snps_only, None)
                        yield "mask", cmask
                        cmask = next(mask_iterator, None)
                    else:
                        yield "mask", cmask
                        cmask = next(mask_iterator, None)

        abnb_miss = [-1, 0, 0] * len(nb)
        abnb_nonseg = sum([[0, 0, x] for x in nb], [])
        multiples = set()
        with RepeatingWriter(out) as rw:
            records = interleaved()
            last_pos = 0
            for ty, rec in records:
                if ty == "mask":
                    span = rec[1] - last_pos
                    rw.write([span] + abnb_nonseg)
                    rw.write([rec[2] - rec[1] + 1] + abnb_miss)
                    last_pos = rec[2]
                    continue
                abnb = rec2gt(rec)
                if rec.pos == last_pos:
                    multiples.add(rec.pos)
                    continue
                span = rec.pos - last_pos - 1
                if 1 <= span <= args.missing_cutoff:
                    rw.write([span] + abnb_nonseg)
                elif span > args.missing_cutoff:
                    rw.write([span] + abnb_miss)
                rw.write([1] + abnb)
                last_pos = rec.pos
            rw.write([contig_length - last_pos] + abnb_nonseg)
        if multiples:
            # FIXME: what to do with multiple records at same site
            logger.warn("Multiple entries found at %d positions; skipped all but the first", len(multiples))
