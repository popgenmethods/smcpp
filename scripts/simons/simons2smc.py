#!/usr/bin/python

import gzip
import sys
import struct
import itertools as it

# read mask files
sn = sys.argv[1]
chr = sys.argv[2]
mask_name = {line.split()[0]: line.split()[2] for line in open("/scratch/simons_masks/sample.masks", "rt")}[sn]
mask_file = "/scratch/simons_masks/%s.mask.fa" % mask_name

mask = []
with open(mask_file, "rt") as f:
    start = ">%s\n" % chr
    for line in f:
        if line == start:
            break
    for line in f:
        if line[0] == ">":
            break
        mask.append(line.strip())
mask = "".join(mask)
print("finished reading %s" % chr)

sample_ind = [line.split()[0] for line in open("/scratch/simons_data_jack/fullypublic.ind", "rt")].index(sn)
snps = open("/scratch/simons_data_jack/fullypublic.%s.snp" % chr, "rt")
geno = open("/scratch/simons_data_jack/fullypublic.%s.geno" % chr, "rt")
variants = {int(snp.split()[3]): int(gt[sample_ind]) for snp, gt in it.izip(snps, geno)}
print("finished reading variants")

seq = [None] * len(mask)
nnm = 0
nv = 0
for i in range(len(mask)):
    if mask[i] in ('N', '0'):
        seq[i] = -1
    else:
        v = variants.get(i, 0)
        if v == 9:
            nnm += 1
            v = -1
        nv += 1
        seq[i] = v
print('num miss/nomiss:', nnm)
print('nv:', nv)
with open("out/%s.%s.seq" % (sn, chr), "wb") as out:
    out.write(struct.pack("%db" % len(seq), *seq))
