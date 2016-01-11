import sys
import bz2
from collections import defaultdict

samples = ["NA%i-200-37-ASM" % i for i in [18501, 18502, 18504, 18505, 18508, 18517, 19129]]
nd = 2 * (len(samples) - 1)
def row2gt(row):
    gts = -1 if 'N' in row[samples[0]] else sum(map(int, row[samples[0]]))
    ungt = [int(x) for s in samples[1:] for g in row[s] for x in g if x != 'N']
    return [gts, sum(ungt), len(ungt)]

chr = defaultdict(lambda: [])
lastpos = defaultdict(lambda: 0)
i = 0
with bz2.BZ2File(sys.argv[1], "rb") as file:
    fields = next(file).strip().split()
    for line in file:
        row = dict(zip(fields, line.strip().split()))
        chrom = row['chromosome']
        c = chr[chrom]
        if c:
            span = int(row['begin']) - int(lastpos[chrom]) - 1
            if span > 0:
                c.append([span, 0, 0, nd])
        c.append([1] + row2gt(row))
        lastpos[chrom] = int(row['begin'])
for c in chr:
    open(c + ".txt", "wt").write("\n".join(["\t".join(map(str, x)) for x in chr[c]]))
