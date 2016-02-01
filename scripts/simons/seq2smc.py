import sys
import os, os.path
import struct
import itertools as it

seqs = []
for f in sys.argv[2:]:
    bytes = open(f, 'rb').read()
    seqs.append(struct.unpack("%db" % len(bytes), bytes))

with open(sys.argv[1], "wt") as out:
    lastobs = None
    span = 0
    for bases in it.izip(*seqs):
        a = max(bases[0], -1)
        undist = [b for b in bases[1:] if b != -1]
        obs = [a, sum(undist), 2 * len(undist)]
        if obs[0] == 2 and obs[1] == obs[2]:
            obs = [0, 0, obs[2]]
        if lastobs is None or obs == lastobs:
            span += 1
        else:
            out.write("%d %d %d %d\n" % tuple([span] + lastobs))
            span = 1
        lastobs = obs
    out.write("%d %d %d %d" % tuple([span] + lastobs))
