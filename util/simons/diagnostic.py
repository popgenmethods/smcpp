#!/usr/bin/env python

import sys
import numpy as np
import pysam
import struct
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WS = 100000

indmap = dict(l.split()[1:3] for l in open("/scratch/simons_masks/sample.masks", "rt"))
ind, chrom = sys.argv[1:3]
fa = pysam.FastaFile("/scratch/simons_masks/%s.mask.fa" % indmap[ind])
mask = fa.fetch(chrom)

bytes = open("/scratch/terhorst/simons/out/%s.%s.seq" % (ind, chrom), "rt").read()
seq = np.array(struct.unpack("%db" % len(bytes), bytes), dtype=float)
seq[seq == -1] = np.nan
avghet = [np.nanmean(seq[i:i+WS]) for i in range(0, len(seq) - 1, WS)]

smc = np.loadtxt("/export/home/terhorst/Dropbox/Berkeley/Research/psmc++/psmcpp/example_datasets/simons/papuan/%s.txt.gz" % chrom, 
        dtype=np.int32)
smcz = np.zeros(smc[:,0].sum(), dtype=float)
s = 0
for row in smc:
    span, a, b, nb = row
    a = float(a) if a > -1 else np.nan
    smcz[s:s+span] = a
    s += span
avgsmc = [np.nanmean(smcz[i:i+WS]) for i in range(0, len(smcz) - 1, WS)]

ary = np.array([float(x) if x != 'N' else np.nan for x in mask])
assert len(seq) == len(ary)
miss = np.isnan(ary)
avgqual = [np.nanmean(ary[i:i+WS]) for i in range(0, len(ary) - 1, WS)]
avgmiss = [np.mean(miss[i:i+WS]) for i in range(0, len(ary) - 1, WS)]

print(seq.shape, smcz.shape, ary.shape, miss.shape)

fig, axes = plt.subplots(nrows=4, sharex=True, figsize=(25, 15))
pos = WS * np.arange(len(avghet))
for ax, obj in zip(axes, [avghet, avgqual, avgmiss, avgsmc]):
    ax.scatter(pos, obj)
    ax.set_xlim((0, len(mask)))
plt.savefig(sys.argv[3])
