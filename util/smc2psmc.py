#!/usr/bin/env python3

import numpy as np
import argparse

from smcpp.estimation_tools import load_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description='Convert SMC++-formatted data set into PSMCfa-style data.')
    parser.add_argument("--contig", help="name of outputted contig")
    parser.add_argument("input", metavar="file.smc[.gz]")
    args = parser.parse_args()
    args.contig = args.contig or args.input
    contig = load_data([args.input])[0]
    L = contig.data[:, 0].sum()
    L += 100 - (L % 100)
    fa = np.full(L, -1)
    last = 0
    for span, a, b, nb in contig.data:
        fa[last:last + span] = a
        last += span
    fa.shape = (L // 100, -1)
    code = fa.max(axis=1).astype('|S1')
    code[code == b'0'] = b'T'
    code[code == b'1'] = b'K'
    code[code == b'2'] = b'T'  # recode monomorphic sites
    code[fa.min(axis=1) == -1] = b'N'
    print(">" + args.contig)
    Lp = len(code) // 79
    if Lp > 0:
        out = np.full([Lp, 80], b"\n", dtype='string_')
        out[:, :-1] = code[:(79 * Lp)].reshape(Lp, 79)
        print(out.tostring().decode('ascii')[:-1])  # omit trailing newline
    print(code[(79 * Lp):].tostring().decode('ascii'))
