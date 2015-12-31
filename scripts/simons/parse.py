import re
import numpy as np

def parse_snps(snp_file):
    sre = re.compile(r' +')
    return np.array([int(sre.split(line.strip())[3]) for line in open(snp_file, "r")])

def parse_geno(geno_file):
    A = np.fromfile(geno_file, dtype='i1')
    assert(A.shape[0] % 284 == 0)
    A.shape = (A.shape[0] // 284, 284)
    # Chop out the newlines
    assert np.all(A[:, -1] == 10)
    A = A[:, :-1]
    A -= 48
    return A

def parse_chrom(base, chrom):
    snp_locs = parse_snps("{base}.{chrom}.snp".format(base=base, chrom=chrom))
    print(snp_locs[:10])
    geno = parse_geno("{base}.{chrom}.geno".format(base=base, chrom=chrom))
    # geno has dimensions locs X ppl
    # get missingness
    missingness = np.mean(geno == 9, axis=0)
    print(missingness)
    return (chrom, snp_locs, geno)

for chrom in [22]: # range(1, 23)[::-1]:
    print(chrom)
    parse_chrom("fullypublic", chrom)
    np.savez_compressed("/scratch/terhorst/simons/{chrom}".format(chrom=chrom), *parse_chrom("fullypublic", chrom))
    print("done")
