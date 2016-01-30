#!/usr/bin/env python
'''Normalize a data set for use in SMC++.'''

import argparse
import numpy as np

import psmcpp._pypsmcpp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("k", type=int, help="emit full SFS only at every <k>th site")
    parser.add_argument("infile", type=str, help="input file")
    parser.add_argument("outfile", type=str, help="output file")
    args = parser.parse_args()

    data = np.loadtxt(args.infile, dtype=np.int32)
    out = psmcpp._pypsmcpp.thin(data, args.k)
    np.savetxt(args.outfile, out)

if __name__ == "__main__":
    main()
