import argparse
import warnings
import itertools as it
from logging import getLogger
import numpy as np
import sys
from pysam import VariantFile, TabixFile
logger = getLogger(__name__)

from ..logging import setup_logging
from ..util import optional_gzip, RepeatingWriter


def init_parser(parser):
    parser.add_argument("--ignore-missing", default=False, action="store_true",
            help="ignore samples which are missing in the data")
    parser.add_argument("-d", "--distinguished",
            nargs="*",
            metavar="sample_id",
            help="ids of the distinguished lineage(s). "
                 "if two values are supplied, values will be pulled "
                 "from the 'left' haplotype of id #1 and 'right' "
                 "haplotype of id #2. (this only makes sense for "
                 "phased data. default: first sample.) ")
    parser.add_argument("--missing-cutoff", "-c", metavar="c", type=int, default=None,
            help="treat runs of homozygosity longer than <c> base pairs as missing")
    parser.add_argument("--mask", "-m",
            help="BED-formatted mask of missing regions",
            widget="FileChooser")
    parser.add_argument("--pop2", "-p", nargs="+",
            metavar="sample_id", default=[],
            help="Column(s) representing second population")
    parser.add_argument("vcf", metavar="vcf[.gz]",
            help="input VCF file", widget="FileChooser")
    parser.add_argument("out", metavar="out[.gz]",
            help="output SMC++ file", widget="FileChooser")
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
    if not args.distinguished:
        args.distinguished = args.sample_ids[:1]
    if len(args.distinguished) == 1:
        args.distinguished *= 2
    dist = [[x for x in args.distinguished if x in pop]
            for pop in (args.sample_ids, args.pop2)]
    undist = [[p for p in pop if p not in args.distinguished]
              for pop in (args.sample_ids, args.pop2)]
    npop = 1
    logger.info("Population 1:")
    logger.info("Distinguished lineages: " + 
                ", ".join(["%s:%d" % c[::-1] for c in enumerate(dist[0], 1)]))
    logger.info("Undistinguished samples: " + ", ".join(undist[0]))
    if args.pop2:
        npop = 2
        common = set(dist[0]) & set(dist[1])
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
        dist = dist[:npop]
        undist = undist[:npop]
        undist = [[sample for sample in u if sample not in missing] for u in undist]
        nb = [2 * len(u) for u in undist]

        # function to convert a VCF record to our format <span, dist gt, undist gt, # undist>
        def rec2gt(rec):
            ref = rec.alleles[0]
            da = [[rec.samples[di].alleles[i] for i, di in enumerate(d)]
                  for d in dist]
            a = [sum(x != ref for x in d) if None not in d else -1
                 for d in dist]
            bs = [[allele != ref for u in und
                   for allele in rec.samples[u].alleles 
                   if allele is not None]
                   for und in undist]
            b = [sum(_) for _ in bs]
            nb = [len(_) for _ in bs]
            ret = list(sum(zip(a, b, nb), tuple()))
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

        abnb_miss = [-1, 0, 0] * len(nb)
        abnb_nonseg = sum([[0, 0, x] for x in nb], [])
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
                span = rec.pos - last_pos - 1
                if 1 <= span <= args.missing_cutoff:
                    rw.write([span] + abnb_nonseg)
                elif span > args.missing_cutoff:
                    rw.write([span] + abnb_miss)
                rw.write([1] + abnb)
                last_pos = rec.pos
            rw.write([contig_length - last_pos] + abnb_nonseg)
