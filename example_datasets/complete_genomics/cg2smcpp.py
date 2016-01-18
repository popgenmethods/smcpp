import sys
import bz2
import gzip
from collections import defaultdict
from itertools import groupby
import multiprocessing
import shelve

samples = ["NA%i-200-37-ASM" % i for i in [18501, 18502, 18504, 18505, 18508, 18517, 19129]]
nd = 2 * (len(samples) - 1)
miss = [-1, 0, 0]
nonseg = [0, 0, nd]

def fasta_iter(fasta_name):
    """
    given a fasta file. yield tuples of header, sequence
    """
    with open(fasta_name, "rt") as fh:
        # ditch the boolean (x[0]) and just keep the header or sequence since
        # we know they alternate.
        faiter = (x[1] for x in groupby(fh, lambda line: line[0] == ">"))
        for header in faiter:
            # drop the ">"
            header = next(header)[1:].strip()
            # join all sequence lines to one.
            seq = "".join(s.strip() for s in next(faiter))
            yield (header, seq)

def process_chrom(args):
    ref, vars, chrom, miss, nonseg = args
    lastob = miss
    span = 1
    j = 1
    allout = gzip.GzipFile("out/%s.txt.gz" % chrom, "w")
    out = gzip.GzipFile("out/%s.%i.txt.gz" % (chrom, j), "w")
    for i, base in enumerate(ref.upper()):
        if base not in "ACTG":
            ob = miss
        else:
            ob = vars.get(i, nonseg)
        if ob == lastob:
            span += 1
        else:
            row = str(span) + "\t" + "\t".join(map(str, lastob)) + "\n"
            allout.write(row)
            if span >= 50000:
                j += 1
                out.close()
                out = gzip.GzipFile("out/%s.%i.txt.gz" % (chrom, j), "w")
            else:
                out.write(row)
            span = 1
        lastob = ob
    out.write(str(span) + "\t" + "\t".join(map(str, lastob)) + "\n")
    allout.close()
    out.close()

def row2gt(row):
    gts = -1 if 'N' in row[samples[0]] else sum(map(int, row[samples[0]]))
    ungt = [int(x) for s in samples[1:] for g in row[s] for x in g if x != 'N']
    return [gts, sum(ungt), len(ungt)]

def main():
    shelf = {}
    if "ref" not in shelf:
        shelf['ref'] = dict(fasta_iter("/scratch/genomic_data/hg19.fa"))
    ref = shelf['ref']

    if 'variants' not in shelf:
        variants = defaultdict(dict)
        with bz2.BZ2File(sys.argv[1], "rb") as file:
            fields = next(file).strip().split()
            for line in file:
                row = dict(zip(fields, line.strip().split()))
                chrom = row['chromosome']
                pos = int(row['begin'])
                variants[chrom][pos] = row2gt(row)
        shelf['variants'] = variants
    variants = shelf['variants']

    p = multiprocessing.Pool(None)
    p.map(process_chrom, [(ref[chrom], variants[chrom], chrom, miss, nonseg) for chrom in variants])
    p.close()
    p.join()

if __name__ == "__main__":
    main()
