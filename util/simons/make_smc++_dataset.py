import numpy as np
import sys
import os.path

MAX_GAP = 10000

base = sys.argv[1]
chrom = sys.argv[2]
ary = np.load("/scratch/terhorst/simons/{chrom}.npz".format(chrom=chrom))
sample_names = [l.strip().split()[0] for l in open("/scratch/terhorst/simons/fullypublic.ind", "rt")]
dist = sample_names.index(sys.argv[3])
ui = [sample_names.index(sn) for sn in sys.argv[4:]]
part = 0
pos = ary['arr_1']
gt = ary['arr_2']
n = 2 * (1 + len(ui))
tmp_ary = []
last_p = None
for i, p in enumerate(pos):
    dist_gt = -1 if gt[i, dist] == 9 else gt[i, dist]
    un = gt[i, ui]
    un_gt = gt[i, ui][gt[i, ui] != 9].sum()
    nb = (un != 9).sum() * 2
    # fold the non-segegrating and cosmopolitan sites
    if dist_gt == 2 and un_gt == nb:
        dist_gt = un_gt = 0
    if last_p is None:
        gap = 0
    else:
        gap = p - last_p - 1
    if gap > 0:
        tmp_ary.append([gap, 0, 0, n - 2])
    tmp_ary.append([1, dist_gt, un_gt, nb])
    last_p = p
tmp_ary2 = []
span = tmp_ary[0][0]
for i in range(1, len(tmp_ary)):
    if tmp_ary[i][1:] == tmp_ary[i - 1][1:]:
        span += tmp_ary[i][0]
    else:
        tmp_ary2.append([span] + tmp_ary[i - 1][1:])
        span = tmp_ary[i][0]
tmp_ary2.append([span] + tmp_ary[-1][1:])
part = 0
i = 0
s = 0
tmp_ary3 = []
while True:
    while i < len(tmp_ary2):
        row = tmp_ary2[i]
        i += 1
        if row[0] > MAX_GAP:
            print(row[0], part)
            if s > 500000:
                np.savetxt(os.path.join(base, "{chrom}.{part}.txt".format(chrom=chrom, part=part)), tmp_ary3, fmt="%d")
            tmp_ary3 = []
            s = 0
            part += 1
            break
        tmp_ary3.append(row)
        s += row[0]
    if i == len(tmp_ary2):
        break
if s > 500000:
    np.savetxt(os.path.join(base, "{chrom}.{part}.txt".format(chrom=chrom, part=part)), tmp_ary3)
