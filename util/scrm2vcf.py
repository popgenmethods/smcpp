#!/usr/bin/env python3
from __future__ import print_function, division
import shutil
import sh
import os
import sys
import argparse
from shutil import which

SCRM = os.environ.get('SCRM_PATH', False) or which('scrm')

if __name__ == "__main__":
    if not SCRM:
        sys.exit("Can't find scrm. Please set SCRM_PATH.")
    scrm = sh.Command(SCRM)
    parser = argparse.ArgumentParser()
    parser.add_argument("--contig", default="contig1", help="name of contig in VCF")
    parser.add_argument("-o", help="output location (default: stdout)")
    parser.add_argument("n", type=int, help="diploid sample size")
    parser.add_argument("rho", type=float, help="recombination rate")
    parser.add_argument("length", type=int, help="length of chromosome to simulate")
    args, scrm_extra_args = parser.parse_known_args()
    if args.o is None:
        out = sys.stdout
    else:
        out = open(args.o, "wt")
    scrm_args = [2 * args.n, 1]
    scrm_args.append("--transpose-segsites")
    scrm_args += ["-SC", "abs"]
    scrm_args += ["-r", args.rho, args.length]
    scrm_args += scrm_extra_args

    # Create a minimal VCF header
    header = ["##fileformat=VCFv4.0", """##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">"""]
    header.append("##contig=<ID={},length={}>".format(args.contig, args.length))
    h = "#CHROM POS ID REF ALT QUAL FILTER INFO FORMAT".split()
    h += ["sample%d" % i for i in range(1, args.n + 1)]
    header.append("\t".join(h))
    print("\n".join(header), file=out)

    # Iterate over scrm output
    it = scrm(*scrm_args, _iter=True)
    line = next(it)
    while not line.startswith("position"):
        line = next(it)
    next(it)
    for line in it:
        pos, time, *gts = line.strip().split()
        cols = [args.contig, str(int(float(pos))), ".", "A", "C", ".", "PASS", ".", "GT"]
        cols += ["/".join(gt) for gt in zip(gts[::2], gts[1::2])]
        print("\t".join(cols), file=out)
    out.close()
