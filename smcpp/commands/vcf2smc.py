import argparse
import warnings
import itertools as it
from logging import getLogger
import numpy as np
import sys
from pysam import VariantFile, TabixFile
import inflect
logger = getLogger(__name__)

from ..logging import setup_logging
from ..util import optional_gzip, RepeatingWriter


def init_parser(parser):
    parser.add_argument("--ignore-missing", default=False, action="store_true",
            help="ignore samples which are missing in the data")
    parser.add_argument("-i", "--distinguished_index", type=int, nargs="*",
            metavar="index",
            help="index of distinguished lineage in sample ids. "
                 "if two values are supplied, values will be pulled "
                 "from the 'left' haplotype of id #1 and 'right' "
                 "haplotype of id #2. this only makes sense for phased data. "
                 "(default: 0)")
    parser.add_argument("--missing-cutoff", "-c", metavar="c", type=int, default=None,
            help="treat runs of homozygosity longer than <c> base pairs as missing")
    parser.add_argument("--mask", "-m", help="BED-formatted mask of missing regions", widget="FileChooser")
    parser.add_argument("--pop2", "-p", nargs="+", metavar="sample_id", default=[],
            help="Column(s) representing second population")
    parser.add_argument("vcf", metavar="vcf[.gz]", help="input VCF file", widget="FileChooser")
    parser.add_argument("out", metavar="out[.gz]", help="output SMC++ file", widget="FileChooser")
    parser.add_argument("contig", help="contig to parse")
    parser.add_argument("sample_ids", nargs="+", metavar="sample_id",
            help="Column(s) to pull from the VCF, or file containing the same.")

def validate(args):
    if args.missing_cutoff and args.mask:
        raise RuntimeError("--missing-cutoff and --mask are mutually exclusive")

def main(args):
    setup_logging(0)
    validate(args)
    if len(args.sample_ids) == 1:
        try:
            with open(args.sample_ids[0], "rt") as f:
                args.sample_ids = [line.strip() for line in f]
        except IOError:
            pass
    if not args.distinguished_index:
        args.distinguished_index = [0]
    if len(args.distinguished_index) == 1:
        args.distinguished_index *= 2
    dist = [args.sample_ids[di] for di in args.distinguished_index]
    undist = [sid for j, sid in enumerate(args.sample_ids) if j not in args.distinguished_index]
    npop = 1
    if args.pop2:
        logger.info("Population 1:")
        npop = 2
    logger.info("Distinguished lineages: " + 
                ", ".join(["%s:%d" % c[::-1] for c in enumerate(dist, 1)]))
    logger.info("Undistinguished samples: " + ", ".join(undist))
    p2 = args.pop2
    if args.pop2:
        logger.info("Population 2:")
        logger.info("Undistinguished samples: " + ", ".join(args.pop2))

    ## Start parsing
    vcf = VariantFile(args.vcf)
    with optional_gzip(args.out, "wt") as out:
        samples = list(vcf.header.samples)
        if not set(dist) <= set(samples):
            raise RuntimeError("Distinguished lineages not found in data?")
        missing = [u for slist in [undist, p2] for u in slist if u not in samples]
        if missing:
            msg = "The following samples were not found in the data: %s. " % ", ".join(missing)
            if args.ignore_missing:
                logger.warn(msg)
            else:
                msg += "If you want to continue without these samples, use --ignore-missing."
                raise RuntimeError(msg)
        undist = [[u for u in slist if u not in missing] for slist in [undist, p2]][:npop]
        nb = [2 * len(u) for u in undist]

        # function to convert a VCF record to our format <span, dist gt, undist gt, # undist>
        def rec2gt(rec):
            ref = rec.alleles[0]
            da = [rec.samples[di].alleles[i] for i, di in enumerate(dist)]
            a = [int(x != ref) if x is not None else -1 for x in da]
            bs = [[allele != ref for u in slist 
                   for allele in rec.samples[u].alleles 
                   if allele is not None]
                   for slist in undist]
            b = [sum(_) for _ in bs]
            nb = [len(_) for _ in bs]
            ret = a + list(sum(zip(b, nb), tuple()))
            return ret

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

        nb_miss = [0, 0] * len(nb)
        nb_nonseg = list(sum(zip([0] * len(nb), nb), tuple()))
        with RepeatingWriter(out) as rw:
            records = interleaved()
            last_pos = 0
            for ty, rec in records:
                if ty == "mask":
                    span = rec[1] - last_pos
                    rw.write([span, 0, 0] + nb_nonseg)
                    rw.write([rec[2] - rec[1] + 1, -1, -1] + nb_miss)
                    last_pos = rec[2]
                    continue
                abnb = rec2gt(rec)
                span = rec.pos - last_pos - 1
                if 1 <= span <= args.missing_cutoff:
                    rw.write([span, 0, 0] + nb_nonseg)
                elif span > args.missing_cutoff:
                    rw.write([span, -1, -1] + nb_miss)
                rw.write([1] + abnb)
                last_pos = rec.pos
            rw.write([contig_length - last_pos, 0, 0] + nb_nonseg)
